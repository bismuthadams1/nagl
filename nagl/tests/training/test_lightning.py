import pathlib

import numpy
import pyarrow
import pyarrow.parquet
import pytest
import torch
import torch.optim
from torch.utils.data import DataLoader

import nagl.nn
import nagl.nn.convolution
import nagl.nn.pooling
import nagl.nn.postprocess
import nagl.nn.readout
from nagl.config import Config, DataConfig, ModelConfig, OptimizerConfig
from nagl.config.data import Dataset, DipoleTarget, ESPTarget, ReadoutTarget
from nagl.config.model import GCNConvolutionModule, ReadoutModule, Sequential
from nagl.datasets import DGLMoleculeDataset
from nagl.features import AtomConnectivity, AtomicElement, BondOrder
from nagl.molecules import DGLMolecule
from nagl.training.lightning import (
    DGLMoleculeDataModule,
    DGLMoleculeLightningModel,
    _hash_featurized_dataset,
)

@pytest.fixture()
def mock_config() -> Config:
    return Config(
        model=ModelConfig(
            atom_features=[AtomConnectivity()],
            bond_features=[],
            convolution=GCNConvolutionModule(
                type="SAGEConv", hidden_feats=[4, 4], activation=["ReLU", "ReLU"]
            ),
            readouts={
                "atom": ReadoutModule(
                    pooling="atom",
                    forward=Sequential(hidden_feats=[2], activation=["Identity"]),
                    postprocess="charges",
                )
            },
        ),
        data=DataConfig(
            training=Dataset(
                sources=[""],
                targets=[
                    ReadoutTarget(column="charges-am1", readout="atom", metric="rmse")
                ],
                batch_size=4,
            ),
            validation=Dataset(
                sources=[""],
                targets=[
                    ReadoutTarget(column="charges-am1", readout="atom", metric="rmse")
                ],
                batch_size=5,
            ),
            test=Dataset(
                sources=[""],
                targets=[
                    ReadoutTarget(column="charges-am1", readout="atom", metric="rmse")
                ],
                batch_size=6,
            ),
        ),
        optimizer=OptimizerConfig(type="Adam", lr=0.01),
    )


@pytest.fixture()
def mock_config_dipole() -> Config:
    return Config(
        model=ModelConfig(
            atom_features=[AtomConnectivity()],
            bond_features=[],
            convolution=GCNConvolutionModule(
                type="SAGEConv", hidden_feats=[4, 4], activation=["ReLU", "ReLU"]
            ),
            readouts={
                "atom": ReadoutModule(
                    pooling="atom",
                    forward=Sequential(hidden_feats=[2], activation=["Identity"]),
                    postprocess="charges",
                )
            },
        ),
        data=DataConfig(
            training=Dataset(
                sources=[""],
                targets=[
                    DipoleTarget(
                        dipole_column="dipole",
                        charge_label="charges-am1",
                        conformation_column="conformation",
                        metric="rmse",
                    )
                ],
                batch_size=4,
            ),
            validation=Dataset(
                sources=[""],
                targets=[
                    DipoleTarget(
                        dipole_column="dipole",
                        charge_label="charges-am1",
                        conformation_column="conformation",
                        metric="rmse",
                    )
                ],
                batch_size=5,
            ),
            test=Dataset(
                sources=[""],
                targets=[
                    DipoleTarget(
                        dipole_column="dipole",
                        charge_label="charges-am1",
                        conformation_column="conformation",
                        metric="rmse",
                    )
                ],
                batch_size=6,
            ),
        ),
        optimizer=OptimizerConfig(type="Adam", lr=0.01),
    )
    
@pytest.fixture()
def mock_config_esp() -> Config:
    from openff.units import unit
    KE = (1 / (4 * numpy.pi * unit.epsilon_0)).m
    return Config(
        model=ModelConfig(
            atom_features=[AtomConnectivity()],
            bond_features=[],
            convolution=GCNConvolutionModule(
                type="SAGEConv", hidden_feats=[4, 4], activation=["ReLU", "ReLU"]
            ),
            readouts={
                "atom": ReadoutModule(
                    pooling="atom",
                    forward=Sequential(hidden_feats=[2], activation=["Identity"]),
                    postprocess="charges",
                )
            },
        ),
        data=DataConfig(
            training=Dataset(
                sources=[""],
                targets=[
                    ESPTarget(
                        esp_column="esp",
                        charge_label="mbis charges",
                        inv_distance_column="inv_distance",
                        metric="rmse",
                        esp_length_column="esp_length",
                        ke = KE,
                    )
                ],
                batch_size=1,
            ),
            validation=Dataset(
                sources=[""],
                targets=[
                    ESPTarget(
                        esp_column="esp",
                        charge_label="mbis charges",
                        inv_distance_column="inv_distance",
                        metric="rmse",
                        esp_length_column="esp_length",
                        ke = KE,
                    )
                ],
                batch_size=1,
            ),
            test=Dataset(
                sources=[""],
                targets=[
                    ESPTarget(
                        esp_column="esp",
                        charge_label="mbis charges",
                        inv_distance_column="inv_distance",
                        metric="rmse",
                        esp_length_column="esp_length",
                        ke = KE,
                    )
                ],
                batch_size=1,
            ),
        ),
        optimizer=OptimizerConfig(type="Adam", lr=0.01),
    )


@pytest.fixture()
def mock_lightning_model(mock_config) -> DGLMoleculeLightningModel:
    return DGLMoleculeLightningModel(mock_config)


def test_hash_featurized_dataset(tmp_cwd):
    labels = pyarrow.table([["C"]], ["smiles"])
    source = str(tmp_cwd / "train.parquet")

    pyarrow.parquet.write_table(labels, source)

    config = Dataset(
        sources=[source],
        targets=[ReadoutTarget(column="label-col", readout="", metric="rmse")],
    )

    atom_features = [AtomicElement()]
    bond_features = [BondOrder()]

    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)

    assert hash_value_1 == hash_value_2

    atom_features = []

    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 != hash_value_2
    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 == hash_value_2

    bond_features = []

    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 != hash_value_2
    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 == hash_value_2

    config.targets[0].column = "label-col-2"

    hash_value_2 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 != hash_value_2
    hash_value_1 = _hash_featurized_dataset(config, atom_features, bond_features)
    assert hash_value_1 == hash_value_2


class TestDGLMoleculeLightningModel:
    def test_init(self, mock_config):
        model = DGLMoleculeLightningModel(mock_config)

        assert isinstance(model.convolution_module, nagl.nn.convolution.SAGEConvStack)
        assert len(model.convolution_module) == 2

        assert all(x in model.readout_modules for x in ["atom"])

        assert isinstance(
            model.readout_modules["atom"].pooling_layer,
            nagl.nn.pooling.AtomPoolingLayer,
        )
        assert isinstance(
            model.readout_modules["atom"].postprocess_layer,
            nagl.nn.postprocess.PartialChargeLayer,
        )

    def test_forward(self, mock_lightning_model, rdkit_methane):
        dgl_molecule = DGLMolecule.from_rdkit(
            rdkit_methane,
            mock_lightning_model.config.model.atom_features,
            mock_lightning_model.config.model.bond_features,
        )

        output = mock_lightning_model.forward(dgl_molecule)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)

    @pytest.mark.parametrize(
        "method_name", ["training_step", "validation_step", "test_step"]
    )
    def test_step_readout(
        self, mock_lightning_model, method_name, dgl_methane, monkeypatch
    ):
        def mock_forward(_):
            return {
                "atom": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
            }

        monkeypatch.setattr(mock_lightning_model, "forward", mock_forward)

        loss = getattr(mock_lightning_model, method_name)(
            (
                dgl_methane,
                {"charges-am1": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])},
            ),
            0,
        )
        assert loss.requires_grad is True
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_step_dipole(self, mock_config_dipole, rdkit_methane, monkeypatch):
        """Make sure the dipole error is correctly calculated and has a gradient"""
        from openff.units import unit

        mock_model = DGLMoleculeLightningModel(mock_config_dipole)

        def mock_forward(_):
            return {
                "charges-am1": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True
                )
            }

        monkeypatch.setattr(mock_model, "forward", mock_forward)
        dgl_methane = DGLMolecule.from_rdkit(
            molecule=rdkit_methane, atom_features=[AtomicElement()]
        )
        # coordinates in angstrom
        conformer = rdkit_methane.GetConformer().GetPositions() * unit.angstrom
        loss = mock_model.training_step(
            (
                dgl_methane,
                {
                    "charges-am1": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
                    "dipole": torch.Tensor([[0.0, 0.0, 0.0]]),
                    "conformation": torch.Tensor([conformer.m_as(unit.bohr)]),
                },
            ),
            0,
        )
        # make sure the gradient is not lost during the calculation
        assert loss.requires_grad is True
        # calculate the loss and compare with numpy
        numpy_dipole = numpy.dot(
            numpy.array([1.0, 2.0, 3.0, 4.0, 5.0]), conformer.m_as(unit.bohr)
        )
        ref_loss = numpy.sqrt(numpy.mean((numpy_dipole - numpy.array([0, 0, 0])) ** 2))
        assert numpy.isclose(loss.detach().numpy(), ref_loss)

    def test_step_esp(self, mock_config_esp, rdkit_methane, rdkit_nitrobromomolecule, monkeypatch):
        """Make sure the esp error is correctly calculated and has a gradient"""
        from openff.units import unit
        KE = 1 / (4 * numpy.pi * unit.epsilon_0)

        mock_model = DGLMoleculeLightningModel(mock_config_esp)
        
        def mock_forward(_):
            return {
                "mbis charges": torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True
                )
            }

        monkeypatch.setattr(mock_model, "forward", mock_forward)
        dgl_methane = DGLMolecule.from_rdkit(
            molecule=rdkit_methane, atom_features=[AtomicElement()]
        )
        # coordinates in angstrom
        loss = mock_model.training_step(
            (
                dgl_methane,
                {
                    "mbis charges": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]]),
                    "esp": torch.Tensor([[0.0, 0.0, 0.0, 1.0, 2.0,0.0, 0.0, 0.0, 1.0, 2.0]]),
                    "inv_distance": torch.Tensor([
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1.]]),
                },
            ),
            0,
        )
       # make sure the gradient is not lost during  the calculation
        assert loss.requires_grad is True

        #test whether we can actually calculate the true esp
    
        mbis_charges = [-0.236,  0.723, -0.389, -0.708,  0.924, -0.602,  0.657, -0.492, 0.447, -0.373,  0.244, -0.769,  0.251,  0.462,  0.466,  0.455, 0.478,  0.461]
        
        esp = [0.228, 0.224, 0.204, 0.242, 0.262, 0.244, 0.248, 0.222, 0.253,
        0.262, 0.276, 0.277, 0.23 , 0.243, 0.23 , 0.192, 0.241, 0.274,
        0.192, 0.234, 0.268, 0.252, 0.247, 0.184, 0.214, 0.226, 0.267,
        0.184, 0.272, 0.213, 0.197, 0.171, 0.234, 0.213, 0.195, 0.175,
        0.248, 0.196, 0.259, 0.199, 0.247, 0.247, 0.189, 0.237, 0.203,
        0.172]
        
        inv_distance = [
        [0.176, 0.203, 0.181, 0.269, 0.307, 0.249, 0.199, 0.166, 0.145,
       0.148, 0.119, 0.376, 0.159, 0.286, 0.162, 0.134, 0.347, 0.393,
       0.188, 0.197, 0.169, 0.244, 0.314, 0.312, 0.237, 0.205, 0.168,
       0.162, 0.137, 0.346, 0.167, 0.229, 0.21 , 0.142, 0.382, 0.293,
       0.149, 0.147, 0.127, 0.179, 0.23 , 0.224, 0.176, 0.153, 0.128,
       0.126, 0.108, 0.302, 0.148, 0.171, 0.156, 0.113, 0.418, 0.276,
       0.2  , 0.261, 0.245, 0.348, 0.311, 0.236, 0.206, 0.169, 0.155,
       0.167, 0.125, 0.312, 0.177, 0.435, 0.156, 0.154, 0.254, 0.342,
       0.242, 0.287, 0.243, 0.371, 0.411, 0.351, 0.286, 0.23 , 0.197,
       0.202, 0.153, 0.365, 0.202, 0.35 , 0.217, 0.176, 0.316, 0.321,
       0.243, 0.238, 0.199, 0.265, 0.325, 0.402, 0.337, 0.309, 0.238,
       0.217, 0.186, 0.28 , 0.201, 0.229, 0.322, 0.182, 0.282, 0.23 ,
       0.196, 0.183, 0.151, 0.22 , 0.313, 0.356, 0.256, 0.217, 0.169,
       0.162, 0.136, 0.367, 0.194, 0.195, 0.224, 0.14 , 0.54 , 0.284,
       0.198, 0.173, 0.146, 0.187, 0.24 , 0.331, 0.282, 0.295, 0.213,
       0.182, 0.177, 0.227, 0.182, 0.163, 0.372, 0.154, 0.262, 0.186,
       0.269, 0.389, 0.38 , 0.418, 0.321, 0.268, 0.261, 0.214, 0.204,
       0.229, 0.158, 0.261, 0.218, 0.437, 0.19 , 0.21 , 0.217, 0.252,
       0.193, 0.24 , 0.216, 0.333, 0.303, 0.215, 0.187, 0.149, 0.137,
       0.152, 0.11 , 0.352, 0.195, 0.422, 0.136, 0.142, 0.273, 0.495,
       0.32 , 0.351, 0.299, 0.338, 0.328, 0.352, 0.372, 0.324, 0.29 ,
       0.292, 0.212, 0.25 , 0.24 , 0.287, 0.286, 0.245, 0.225, 0.218,
       0.27 , 0.238, 0.206, 0.225, 0.244, 0.314, 0.358, 0.44 , 0.359,
       0.284, 0.282, 0.199, 0.214, 0.192, 0.464, 0.233, 0.195, 0.171,
       0.192, 0.2  , 0.169, 0.257, 0.305, 0.246, 0.203, 0.162, 0.142,
       0.151, 0.113, 0.387, 0.206, 0.249, 0.152, 0.136, 0.367, 0.408,
       0.197, 0.169, 0.151, 0.161, 0.176, 0.222, 0.247, 0.338, 0.293,
       0.218, 0.303, 0.153, 0.171, 0.142, 0.445, 0.188, 0.156, 0.134,
       0.208, 0.194, 0.16 , 0.234, 0.311, 0.307, 0.243, 0.197, 0.163,
       0.165, 0.129, 0.355, 0.224, 0.208, 0.19 , 0.145, 0.408, 0.299,
       0.232, 0.332, 0.451, 0.287, 0.216, 0.19 , 0.2  , 0.174, 0.182,
       0.216, 0.149, 0.183, 0.198, 0.311, 0.154, 0.217, 0.158, 0.183,
       0.309, 0.39 , 0.439, 0.299, 0.242, 0.236, 0.27 , 0.245, 0.266,
       0.314, 0.206, 0.192, 0.236, 0.275, 0.209, 0.3  , 0.169, 0.179,
       0.309, 0.288, 0.273, 0.234, 0.22 , 0.25 , 0.312, 0.348, 0.407,
       0.383, 0.317, 0.176, 0.233, 0.204, 0.299, 0.334, 0.163, 0.158,
       0.172, 0.159, 0.157, 0.141, 0.139, 0.156, 0.178, 0.211, 0.24 ,
       0.208, 0.293, 0.121, 0.147, 0.13 , 0.211, 0.202, 0.117, 0.111,
       0.224, 0.254, 0.218, 0.322, 0.308, 0.233, 0.21 , 0.164, 0.151,
       0.17 , 0.119, 0.319, 0.249, 0.329, 0.148, 0.157, 0.262, 0.352,
       0.29 , 0.277, 0.217, 0.34 , 0.404, 0.344, 0.297, 0.219, 0.189,
       0.207, 0.142, 0.375, 0.342, 0.287, 0.195, 0.182, 0.331, 0.329,
       0.291, 0.232, 0.184, 0.253, 0.321, 0.391, 0.356, 0.284, 0.225,
       0.224, 0.167, 0.284, 0.334, 0.209, 0.262, 0.188, 0.292, 0.233,
       0.193, 0.156, 0.14 , 0.145, 0.158, 0.2  , 0.232, 0.336, 0.31 ,
       0.221, 0.374, 0.138, 0.177, 0.128, 0.441, 0.192, 0.141, 0.122,
       0.228, 0.277, 0.373, 0.215, 0.175, 0.167, 0.189, 0.178, 0.203,
       0.245, 0.174, 0.147, 0.194, 0.212, 0.155, 0.271, 0.131, 0.143,
       0.248, 0.357, 0.422, 0.335, 0.236, 0.191, 0.196, 0.16 , 0.163,
       0.201, 0.129, 0.205, 0.251, 0.387, 0.14 , 0.202, 0.171, 0.216,
       0.256, 0.262, 0.296, 0.201, 0.176, 0.183, 0.22 , 0.227, 0.286,
       0.328, 0.256, 0.145, 0.208, 0.185, 0.197, 0.366, 0.133, 0.136,
       0.34 , 0.366, 0.299, 0.376, 0.318, 0.264, 0.27 , 0.205, 0.195,
       0.238, 0.146, 0.265, 0.441, 0.33 , 0.175, 0.219, 0.222, 0.256,
       0.14 , 0.122, 0.118, 0.111, 0.112, 0.129, 0.147, 0.182, 0.204,
       0.168, 0.312, 0.1  , 0.129, 0.101, 0.193, 0.164, 0.1  , 0.092,
       0.342, 0.233, 0.19 , 0.218, 0.242, 0.309, 0.381, 0.374, 0.317,
       0.301, 0.227, 0.201, 0.409, 0.18 , 0.323, 0.246, 0.198, 0.172,
       0.159, 0.132, 0.124, 0.121, 0.126, 0.149, 0.172, 0.226, 0.246,
       0.19 , 0.382, 0.111, 0.149, 0.108, 0.249, 0.178, 0.112, 0.101,
       0.167, 0.134, 0.122, 0.125, 0.134, 0.163, 0.186, 0.243, 0.242,
       0.189, 0.296, 0.12 , 0.162, 0.111, 0.275, 0.172, 0.122, 0.107,
       0.221, 0.271, 0.392, 0.208, 0.164, 0.153, 0.172, 0.158, 0.179,
       0.227, 0.153, 0.14 , 0.204, 0.208, 0.137, 0.262, 0.124, 0.139,
       0.284, 0.299, 0.384, 0.212, 0.175, 0.175, 0.212, 0.205, 0.257,
       0.351, 0.217, 0.144, 0.247, 0.197, 0.173, 0.478, 0.13 , 0.137,
       0.273, 0.317, 0.33 , 0.272, 0.215, 0.188, 0.203, 0.169, 0.176,
       0.223, 0.139, 0.184, 0.321, 0.265, 0.145, 0.227, 0.159, 0.184,
       0.167, 0.151, 0.154, 0.128, 0.122, 0.134, 0.158, 0.184, 0.234,
       0.218, 0.328, 0.106, 0.15 , 0.118, 0.174, 0.238, 0.102, 0.099,
       0.139, 0.124, 0.123, 0.109, 0.107, 0.12 , 0.138, 0.164, 0.195,
       0.172, 0.293, 0.095, 0.128, 0.101, 0.163, 0.178, 0.093, 0.088,
       0.275, 0.194, 0.172, 0.17 , 0.176, 0.211, 0.268, 0.319, 0.352,
       0.313, 0.3  , 0.149, 0.302, 0.146, 0.282, 0.279, 0.146, 0.133,
       0.185, 0.144, 0.131, 0.132, 0.139, 0.165, 0.194, 0.239, 0.25 ,
       0.209, 0.273, 0.123, 0.191, 0.116, 0.243, 0.194, 0.123, 0.11 ,
       0.271, 0.241, 0.264, 0.18 , 0.158, 0.164, 0.202, 0.204, 0.269,
       0.363, 0.239, 0.131, 0.265, 0.163, 0.173, 0.565, 0.121, 0.124,
       0.186, 0.165, 0.173, 0.135, 0.126, 0.135, 0.162, 0.182, 0.242,
       0.251, 0.298, 0.108, 0.172, 0.124, 0.165, 0.307, 0.102, 0.101,
       0.316, 0.255, 0.252, 0.196, 0.175, 0.182, 0.225, 0.217, 0.263,
       0.354, 0.213, 0.146, 0.374, 0.174, 0.18 , 0.425, 0.134, 0.136,
       0.239, 0.193, 0.194, 0.155, 0.145, 0.159, 0.2  , 0.226, 0.321,
       0.348, 0.357, 0.122, 0.231, 0.138, 0.196, 0.448, 0.116, 0.113,
       0.156, 0.136, 0.137, 0.116, 0.113, 0.124, 0.147, 0.172, 0.218,
       0.202, 0.323, 0.099, 0.147, 0.107, 0.164, 0.221, 0.095, 0.092,
       0.267, 0.2  , 0.189, 0.164, 0.158, 0.176, 0.222, 0.243, 0.311,
       0.34 , 0.283, 0.133, 0.294, 0.144, 0.208, 0.368, 0.127, 0.122,
       0.196, 0.154, 0.144, 0.134, 0.134, 0.154, 0.186, 0.221, 0.265,
       0.24 , 0.305, 0.116, 0.204, 0.119, 0.206, 0.241, 0.114, 0.106,
       0.142, 0.12 , 0.115, 0.107, 0.107, 0.122, 0.141, 0.17 , 0.198,
       0.172, 0.283, 0.096, 0.139, 0.097, 0.17 , 0.175, 0.095, 0.088]
        ]
    
        def mock_forward(_):
                return {
                    "mbis charges": torch.tensor(
                        [mbis_charges], requires_grad=True
                    )
                }
                
        monkeypatch.setattr(mock_model, "forward", mock_forward)
        dgl_nitrobromolecule = DGLMolecule.from_rdkit(
            molecule=rdkit_nitrobromomolecule, atom_features=[AtomicElement()]
        )

        # coordinates in angstrom
        loss = mock_model.training_step(
            (
                dgl_nitrobromolecule,
                {
                    "mbis charges": torch.tensor([[mbis_charges]]),
                    "esp": torch.Tensor(esp),
                    "inv_distance": torch.Tensor(inv_distance),
                },
            ),
            0,
        )
        #calculate the numpy version
        numpy_esp = KE.m * (
            numpy.array(inv_distance).reshape(-1,len(numpy.array(mbis_charges)))
            @ numpy.array(mbis_charges)
        )
        
        ref_loss = numpy.sqrt(numpy.mean((numpy_esp - numpy.array(esp)) ** 2))
         
        assert numpy.isclose(loss.detach().numpy(), ref_loss)

    def test_configure_optimizers(self, mock_lightning_model):
        optimizer = mock_lightning_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(optimizer.defaults["lr"]), torch.tensor(0.01))

    def test_yaml_round_trip(self, tmp_cwd, mock_config):
        """Test writing a model to yaml and reloading"""
        model_a = DGLMoleculeLightningModel(mock_config)
        file_name = pathlib.Path("model_a.yaml")
        model_a.to_yaml(file_name)
        model_b = DGLMoleculeLightningModel.from_yaml(file_name)
        assert model_b.hparams["config"] == model_a.hparams["config"]


class TestDGLMoleculeDataModule:
    def test_init(self, tmp_cwd, mock_config):
        data_module = DGLMoleculeDataModule(mock_config, cache_dir=tmp_cwd / "cache")

        for stage in ["train", "val", "test"]:
            assert stage in data_module._data_set_configs
            assert stage in data_module._data_set_paths

            assert callable(getattr(data_module, f"{stage}_dataloader"))

        data_module._data_sets["train"] = DGLMoleculeDataset([])

        loader = data_module.train_dataloader()
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 4

    def test_prepare(self, tmp_cwd, mock_config, mocker):
        mocker.patch(
            "nagl.training.lightning._hash_featurized_dataset",
            autospec=True,
            return_value="hash-val",
        )

        parquet_path = tmp_cwd / "unfeaturized.parquet"
        pyarrow.parquet.write_table(
            pyarrow.table(
                [
                    ["[O-:1][H:2]", "[H:1][H:2]"],
                    [numpy.arange(2).astype(float), numpy.zeros(2).astype(float)],
                    [numpy.arange(2).astype(float) + 2, None],
                ],
                ["smiles", "charges-am1", "charges-am1bcc"],
            ),
            parquet_path,
        )

        mock_config.data.validation = None
        mock_config.data.test = None
        mock_config.data.training.sources = [str(parquet_path)]

        data_module = DGLMoleculeDataModule(mock_config, cache_dir=tmp_cwd / "cache")
        data_module.prepare_data()

        expected_path = tmp_cwd / "cache" / "train-hash-val.parquet"
        assert expected_path.is_file()

        table = pyarrow.parquet.read_table(expected_path)
        assert len(table) == 2

        del data_module._data_sets["train"]
        assert "train" not in data_module._data_sets
        data_module.setup("train")
        assert isinstance(data_module._data_sets["train"], DGLMoleculeDataset)
        assert len(data_module._data_sets["train"]) == 2
