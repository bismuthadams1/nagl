import dataclasses
import functools
import hashlib
import json
import logging
import pathlib
import subprocess
import tempfile
import typing
import os

import pyarrow.parquet
import pydantic
import pytorch_lightning as pl
import rich.progress
import torch
import torch.nn
import yaml
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

import nagl.nn
import nagl.nn.convolution
import nagl.nn.pooling
import nagl.nn.postprocess
import nagl.nn.readout
from nagl.config import Config
from nagl.config.data import Dataset as DatasetConfig
from nagl.config.model import ActivationFunction
from nagl.datasets import DGLMoleculeDataset, collate_dgl_molecules
from nagl.features import AtomFeature, BondFeature
from nagl.molecules import DGLMolecule, DGLMoleculeBatch, MoleculeToDGLFunc
from nagl.training.loss import get_loss_function

_BatchType = typing.Tuple[
    typing.Union[DGLMolecule, DGLMoleculeBatch], typing.Dict[str, torch.Tensor]
]

_logger = logging.getLogger(__name__)


def _get_activation(
    types: typing.Optional[typing.List[ActivationFunction]],
) -> typing.Optional[typing.List[torch.nn.Module]]:
    return (
        None
        if types is None
        else [nagl.nn.get_activation_func(type_)() for type_ in types]
    )


def _hash_featurized_dataset(
    dataset_config: DatasetConfig,
    atom_features: typing.List[AtomFeature],
    bond_features: typing.List[BondFeature],
) -> str:
    """A quick and dirty way to hash a 'featurized' dataset.

    Args:
        dataset_config: The dataset configuration.
        atom_features: The atom feature set.
        bond_features: The bond feature set.

    Returns:
        The dataset hash.
    """

    @pydantic.dataclasses.dataclass
    class DatasetHash:
        atom_features: typing.List[AtomFeature]
        bond_features: typing.List[BondFeature]

        columns: typing.List[str]
        source_hashes: typing.List[str]

    source_hashes = []

    if dataset_config.sources is not None:
        for source in dataset_config.sources:
            result = subprocess.run(
                ["openssl", "sha256", source], capture_output=True, text=True
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"openssl failed: {pathlib.Path.cwd()} {result.stdout} {result.stderr}"
                )

            result.check_returncode()

            source_hashes.append(result.stdout)

    columns = _get_target_columns(targets=dataset_config.targets)

    dataset_hash = DatasetHash(
        atom_features=atom_features,
        bond_features=bond_features,
        source_hashes=source_hashes,
        columns=columns,
    )
    dataset_hash_json = json.dumps(dataclasses.asdict(dataset_hash), sort_keys=True)

    return hashlib.sha256(dataset_hash_json.encode()).hexdigest()


def _get_target_columns(targets) -> typing.List[str]:
    """Get a list of unique column names to extract from the dataset based on if column is in the name"""
    columns = set()
    for target in targets:
        for field, value in target.__dict__.items():
            if "column" in field:
                columns.add(value)
    return sorted(columns)


class DGLMoleculeLightningModel(pl.LightningModule):
    """A model which applies a graph convolutional step followed by multiple (labelled)
    pooling and readout steps.
    """

    def __init__(self, config: typing.Union[Config, typing.Dict[str, typing.Any]]):
        super().__init__()
        if not isinstance(config, Config):
            config = Config(**config)

        self.save_hyperparameters({"config": dataclasses.asdict(config)})

        self.config = config

        n_input_feats = sum(len(feature) for feature in self.config.model.atom_features)

        convolution_class = nagl.nn.convolution.get_convolution_layer(
            self.config.model.convolution.type
        )
        self.convolution_module = convolution_class(
            n_input_feats,
            self.config.model.convolution.hidden_feats,
            _get_activation(self.config.model.convolution.activation),
            self.config.model.convolution.dropout,
        )
        self.readout_modules = torch.nn.ModuleDict(
            {
                readout_name: nagl.nn.readout.ReadoutModule(
                    pooling_layer=nagl.nn.pooling.get_pooling_layer(
                        readout_config.pooling
                    )(),
                    forward_layers=nagl.nn.Sequential(
                        self.config.model.convolution.hidden_feats[-1],
                        readout_config.forward.hidden_feats,
                        _get_activation(readout_config.forward.activation),
                        readout_config.forward.dropout,
                    ),
                    postprocess_layer=nagl.nn.postprocess.get_postprocess_layer(
                        readout_config.postprocess
                    )(),
                )
                for readout_name, readout_config in self.config.model.readouts.items()
            }
        )

    def forward(
        self, molecule: typing.Union[DGLMolecule, DGLMoleculeBatch]
    ) -> typing.Dict[str, torch.Tensor]:
        molecule.graph.ndata["h"] = self.convolution_module(
            molecule.graph, molecule.atom_features
        )
        readouts: typing.Dict[str, torch.Tensor] = {
            readout_type: readout_module(molecule)
            for readout_type, readout_module in self.readout_modules.items()
        }

        return readouts

    def to_yaml(self, path: pathlib.Path):
        """Export the model config to a yaml file"""

        with open(path, "w") as f:
            yaml.dump(self.hparams["config"], f)

    @classmethod
    def from_yaml(cls, path: pathlib.Path):
        """Load the model from a yaml file containing the config"""

        dct = yaml.safe_load(path.read_text())
        return cls(config=dct)

    def _default_step(
        self,
        batch: _BatchType,
        step_type: typing.Literal["train", "val", "test"],
    ):
        molecule, labels = batch

        dataset_configs = {
            "train": self.config.data.training,
            "val": self.config.data.validation,
            "test": self.config.data.test,
        }
        targets = [
            get_loss_function(target.__class__.__name__)(**dataclasses.asdict(target))
            for target in dataset_configs[step_type].targets
        ]

        y_pred = self.forward(molecule)
        metric = torch.zeros(1).type_as(next(iter(y_pred.values())))

        for target in targets:
            if labels[target.target_column()] is None:
                continue

            target_metric = target.evaluate_loss(
                labels=labels, prediction=y_pred, molecules=molecule
            )
            self.log(
                f"{step_type}/{target.target_column()}/{target.metric}/{target.weight}/{target.denominator}",
                target_metric,
            )

            metric += target_metric

        self.log(f"{step_type}/loss", metric)
        return metric

    def training_step(self, train_batch, batch_idx):
        return self._default_step(train_batch, "train")

    def validation_step(self, val_batch, batch_idx):
        return self._default_step(val_batch, "val")

    def test_step(self, test_batch, batch_idx):
        metric = self._default_step(test_batch, "test")

        if isinstance(self.logger, MLFlowLogger):
            self._log_report_artifact(test_batch)

        return metric

    def _log_report_artifact(self, batch_and_labels: _BatchType):
        batch, labels = batch_and_labels

        prediction = self.forward(batch)

        targets = [
            get_loss_function(target.__class__.__name__)(**dataclasses.asdict(target))
            for target in self.config.data.test.targets
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = pathlib.Path(tmp_dir)

            for target in targets:
                if labels[target.target_column()] is None:
                    continue

                report_path = target.report_artifact(
                    molecules=batch,
                    labels=labels,
                    prediction=prediction,
                    output_folder=tmp_dir,
                )

                self.logger.experiment.log_artifact(
                    self.logger.run_id, local_path=str(report_path)
                )

    def configure_optimizers(self):
        if self.config.optimizer.type.lower() != "adam":
            raise NotImplementedError

        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.optimizer.lr)
        return optimizer


class DGLMoleculeDataModule(pl.LightningDataModule):
    """A utility class that makes loading and featurizing train, validation and test
    sets more compact."""

    def __init__(
        self,
        config: Config,
        molecule_to_dgl: typing.Optional[MoleculeToDGLFunc] = None,
        cache_dir: typing.Optional[pathlib.Path] = None,
        n_workers: int = 0,
        progress_bar: bool = True,
    ):
        """

        Args:
            config: The configuration defining what data should be included.
            cache_dir: The (optional) directory to store and load cached featurized data
                in. **No validation is done to ensure the loaded data matches the input
                config so be extra careful when using this option**.
            n_workers: The number of workers to distribute the data set preparation /
                setup over. A value of 0 indicates no multiprocessing should be used.
            molecule_to_dgl: A (optional) callable to use when converting an RDKit
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_rdkit`` class method is used.
            progress_bar: Whether to show a progress bar when preparing / setting up the
                data.
        """
        super().__init__()

        self._config = config
        self._cache_dir = cache_dir

        self._n_workers = n_workers
        self._molecule_to_dgl = molecule_to_dgl

        self._progress_bar = progress_bar

        self._data_sets: typing.Dict[str, DGLMoleculeDataset] = {}

        data_set_configs = {
            "train": config.data.training,
            "val": config.data.validation,
            "test": config.data.test,
        }

        self._data_set_configs: typing.Dict[str, DatasetConfig] = {
            k: v for k, v in data_set_configs.items() if v is not None
        }
        self._data_set_paths = {
            stage: dataset_config.sources
            for stage, dataset_config in self._data_set_configs.items()
        }

        for stage, dataset_config in self._data_set_configs.items():
            self._create_dataloader(dataset_config, stage)

    def _create_dataloader(
        self,
        dataset_config: DatasetConfig,
        stage: typing.Literal["train", "val", "test"],
    ):
        def _factory() -> DataLoader:
            target_data = self._data_sets[stage]

            batch_size = (
                len(target_data)
                if dataset_config.batch_size is None
                else dataset_config.batch_size
            )
            cpus_for_dataloader = os.cpu_count()
            msg = f"number of cpus available for dataloader: {cpus_for_dataloader}"
            logging.info(msg)
            return DataLoader(
                dataset=target_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=cpus_for_dataloader,   #modify this 
                collate_fn=collate_dgl_molecules,
            )

        setattr(self, f"{stage}_dataloader", _factory)

    def prepare_data(self):
        for stage, stage_paths in self._data_set_paths.items():
            _logger.info(f"preparing {stage}")

            dataset_config = self._data_set_configs[stage]
            columns = _get_target_columns(targets=dataset_config.targets)

            hash_string = (
                None
                if self._cache_dir is None
                else _hash_featurized_dataset(
                    dataset_config,
                    self._config.model.atom_features,
                    self._config.model.bond_features,
                )
            )
            cached_path: pathlib.Path = (
                None
                if self._cache_dir is None
                else self._cache_dir / f"{stage}-{hash_string}.parquet"
            )

            if self._cache_dir is not None and cached_path.is_file():
                _logger.info(f"found cached featurized dataset at {cached_path}")
                continue

            progress_bar = functools.partial(
                rich.progress.track, description=f"featurizing {stage} set"
            )

            dataset = DGLMoleculeDataset.from_unfeaturized(
                [pathlib.Path(path) for path in stage_paths],
                columns=columns,
                atom_features=self._config.model.atom_features,
                bond_features=self._config.model.bond_features,
                molecule_to_dgl=self._molecule_to_dgl,
                progress_iterator=None if not self._progress_bar else progress_bar,
                n_processes=self._n_workers,
            )

            self._data_sets[stage] = dataset

            if self._cache_dir is not None:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                pyarrow.parquet.write_table(dataset.to_table(), cached_path)

    def setup(self, stage: typing.Optional[str] = None):
        if self._cache_dir is None:
            return

        for stage in self._data_set_paths:
            if stage in self._data_sets and self._data_sets[stage] is not None:
                continue

            hash_string = _hash_featurized_dataset(
                self._data_set_configs[stage],
                self._config.model.atom_features,
                self._config.model.bond_features,
            )

            progress_bar = functools.partial(
                rich.progress.track, description=f"loading cached {stage} set"
            )

            self._data_sets[stage] = DGLMoleculeDataset.from_featurized(
                self._cache_dir / f"{stage}-{hash_string}.parquet",
                columns=None,
                progress_iterator=None if not self._progress_bar else progress_bar,
                n_processes=self._n_workers,
            )
