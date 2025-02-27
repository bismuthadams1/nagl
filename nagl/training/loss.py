"""Classes to calculate per-readout or global loss functions"""
import abc
import pathlib
import typing
import itertools

import torch

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.training.metrics import MetricType, get_metric


class _BaseTarget(abc.ABC):
    """A general target class used to evaluate the loss of a model"""

    def __init__(self, metric: MetricType, denominator: float, weight: float):
        self.metric = metric
        self.denominator = denominator
        self.weight = weight

    @abc.abstractmethod
    def evaluate_loss(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Evaluate the loss for this target type.

        Notes:
            If molecules is a batch the labels and predictions will be contiguous arrays which will need
            to be split for molecular based errors, this is not needed for atomic based errors.

        Args:
            molecules: The molecule or batch of molecules the labels and predictions correspond to
            labels: A dictionary of labels and reference values for this molecule or batch of molecules
            prediction: A dictionary of labels and predictions made by the model for this molecule or batch of molecules

        Returns:
            The loss calculated by the given metric function
        """
        ...

    @abc.abstractmethod
    def target_column(self) -> str:
        """The name of the column in the data we are calculating the loss against"""
        ...

    @abc.abstractmethod
    def report_artifact(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
        output_folder: pathlib.Path,
    ):
        """
        Create an artifact for this loss metric which can be viewed by MLFLow.

        Args:
            molecules: The batch or molecules that should be included in the report
            labels: A dictionary of labels and reference values for this molecule or batch of molecules
            prediction: A dictionary of labels and predicted values made by the model for this molecule or batch of molecules
            output_folder: The path the folder artifact should be saved to

        Returns:
            The path to the artifact
        """
        ...


class ReadoutTarget(_BaseTarget):
    """A basic loss function acting on a single readout property"""

    def __init__(
        self,
        metric: MetricType,
        denominator: float,
        weight: float,
        column: str,
        readout: str,
    ):
        super().__init__(metric, denominator, weight)
        self.readout = readout
        self.column = column

    def evaluate_loss(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        metric_func = get_metric(self.metric)
        target_labels = labels[self.column]
        target_y_pred = prediction[self.readout]
        return (
            metric_func(target_y_pred, target_labels) * self.weight / self.denominator
        )

    def target_column(self) -> str:
        return self.column

    def report_artifact(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
        output_folder: pathlib.Path,
    ) -> pathlib.Path:
        from nagl.reporting import create_atom_label_report

        # break up the molecules, predictions and labels and pass to the error function one at a time
        if isinstance(molecules, DGLMoleculeBatch):
            molecules = molecules.unbatch()
        else:
            molecules = [molecules]

        n_atoms_per_mol = [molecule.n_atoms for molecule in molecules]

        predictions = torch.split(prediction[self.readout], n_atoms_per_mol)
        labels = torch.split(labels[self.column], n_atoms_per_mol)

        report_entries = [
            (molecule, predictions[i], labels[i])
            for i, molecule in enumerate(molecules)
        ]

        report_path = output_folder.joinpath(f"{self.column}.html")

        create_atom_label_report(
            report_entries,
            metrics=[self.metric],
            rank_by=self.metric,
            output_path=report_path,
        )

        return report_path


class DipoleTarget(_BaseTarget):
    """Calculate the molecular dipole loss based on some predicted point charges stored in the charge_label.

    Calculate the supplied metric between the dipole vectors.
    """

    def __init__(
        self,
        metric: MetricType,
        denominator: float,
        weight: float,
        dipole_column: str,
        conformation_column: str,
        charge_label: str,
    ):
        super().__init__(metric, denominator, weight)
        self.dipole_column = dipole_column
        self.conformation_column = conformation_column
        self.charge_label = charge_label

    def evaluate_loss(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        metric_func = get_metric(self.metric)
        # reshape as it can be flat
        target_dipole = torch.reshape(labels[self.dipole_column], (-1, 3))
        n_atoms_per_molecule = (
            (molecules.n_atoms,)
            if isinstance(molecules, DGLMolecule)
            else molecules.n_atoms_per_molecule
        )
        # reshape the array incase it is flat
        conformation = torch.reshape(labels[self.conformation_column], (-1, 3))

        # split the total array by the number of atoms per molecule
        charges = torch.split(
            prediction[self.charge_label].squeeze(), n_atoms_per_molecule
        )
        conformations = torch.split(conformation, n_atoms_per_molecule)

        predicted_dipoles = torch.stack(
            [
                torch.matmul(charge_slice, conformation_slice)
                for charge_slice, conformation_slice in zip(charges, conformations)
            ]
        )

        # get the error across all dipoles
        return (
            metric_func(predicted_dipoles, target_dipole)
            * self.weight
            / self.denominator
        )

    def target_column(self) -> str:
        return self.dipole_column

    def report_artifact(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
        output_folder: pathlib.Path,
    ):
        from nagl.reporting import create_molecule_label_report

        # break up the molecules, predictions and labels and pass to the error function one at a time
        if isinstance(molecules, DGLMoleculeBatch):
            molecules = molecules.unbatch()
        else:
            molecules = [molecules]

        n_atoms_per_mol = [molecule.n_atoms for molecule in molecules]

        # reshape the array incase it is flat to make splitting work
        conformations = torch.reshape(labels[self.conformation_column], (-1, 3))
        # reshape to help with iterating over molecules
        target_dipole = torch.reshape(labels[self.dipole_column], (-1, 3))

        charges = torch.split(prediction[self.charge_label].squeeze(), n_atoms_per_mol)
        conformations = torch.split(conformations, n_atoms_per_mol)

        entries_and_metrics = []
        # calculate the error a molecule at a time
        for i, molecule in enumerate(molecules):
            error = self.evaluate_loss(
                molecules=molecule,
                labels={
                    self.conformation_column: conformations[i],
                    self.dipole_column: target_dipole[i],
                },
                prediction={self.charge_label: charges[i]},
            )
            # remove the weighting and denominator to get the raw output
            entries_and_metrics.append(
                (molecule, error * self.denominator / self.weight)
            )

        report_path = output_folder.joinpath(f"{self.dipole_column}.html")
        create_molecule_label_report(
            entries_and_metrics=entries_and_metrics,
            metric_label=self.metric,
            output_path=report_path,
        )
        return report_path



class ESPTarget(_BaseTarget):
    """Calculate the molecular ESP loss based on some predicted charges stored in the charge_label.

    Calculate the supplied metric between the ESP vectors.
    """

    def __init__(
        self,
        metric: MetricType,
        denominator: float,
        weight: float,
        esp_column: str,
        inv_distance_column: str,
        esp_length_column: str,
        charge_label: str,
        ke : float,
    ):
        """Initialize the ESPTarget class
        metric: MetricType
            the type of metric, rmse, mse, or mae used to evaluate the loss
        denominator: float
            scales the final loss value
        weight: float
            weights associated with the edges
        esp_column: str
            column key for the esp data in the labels dictionary
        inv_distance_column: str
            distance between the conformer points and grid points
        charge_label: str
            key for charges in the labels dictionary
        ke: float
            Coulombs constant
        """
        super().__init__(metric, denominator, weight)
        self.esp_column = esp_column
        self.charge_label = charge_label
        self.inv_distance_column = inv_distance_column
        self.esp_length_column = esp_length_column
        self.ke = ke
    
    def evaluate_loss(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        metric_func = get_metric(self.metric)
        # split esp by the supplied length column, if batched, this column should be a list. 

        #need to chain the list of ints and lists as batched records will have list of lengths
        print('esp length column')
        print(labels[self.esp_length_column])
        esp_lengths = list(
            itertools.chain.from_iterable(
                [item] if isinstance(item, int) 
                else item for item in labels[self.esp_length_column]
            )
        )
        # print('processed esp lengths')
        # print(esp_lengths)

        #here we grab the number of atoms per molecule
        n_atoms_per_molecule = (
            (molecules.n_atoms,)
            if isinstance(molecules, DGLMolecule)
            else molecules.n_atoms_per_molecule
        )
        #split the esp columns by their length
        target_esps = torch.cat(
        torch.split(
            labels[self.esp_column],
            esp_lengths)
        ).flatten()
        
        # print('target esps')
        # print(target_esps)
        # print('target esps shape')
        # print(target_esps.shape)

        #work out the inv distance chunks, these will be esp length * num_atoms
        inv_distance_chunks = [
            n_atoms * esp_len for n_atoms, esp_len in zip(n_atoms_per_molecule, esp_lengths) 
        ]

        #split the inv distances by these chunks
        inv_distances = torch.split(
            labels[self.inv_distance_column],inv_distance_chunks
        )
        
        # split the total array by the number of atoms per molecule
        charges = torch.split(
            prediction[self.charge_label].squeeze(), n_atoms_per_molecule
        )

        predicted_esps = torch.cat(
        [
            self.ke * (inv_distance_item.reshape(-1,len(charge_slice)) @ charge_slice)
            for charge_slice, inv_distance_item in zip(charges, inv_distances)
        ]
        )
        
        # print('predicted esps')
        # print(predicted_esps)
        # print('predicted_esps shape')
        # print(predicted_esps.shape)
        # get the error across all dipoles
        return (
            metric_func(predicted_esps, target_esps)
            * self.weight
            / self.denominator
        )
        
    def target_column(self) -> str:
        return self.esp_column
    
    def report_artifact(
        self,
        molecules: typing.Union[DGLMolecule, DGLMoleculeBatch],
        labels: typing.Dict[str, torch.Tensor],
        prediction: typing.Dict[str, torch.Tensor],
        output_folder: pathlib.Path,
    ):
        from nagl.reporting import create_molecule_label_report
        
        # break up the molecules, predictions and labels and pass to the error function one at a time
        if isinstance(molecules, DGLMoleculeBatch):
            molecules = molecules.unbatch()
        else:
            molecules = [molecules]
          
        n_atoms_per_mol = [molecule.n_atoms for molecule in molecules]
        #need to chain the list of ints and lists as batched records will have list of lengths
        esp_lengths = list(
            itertools.chain.from_iterable([item] if isinstance(item, int) else item for item in labels[self.esp_length_column])
        )
        # print('esps lengths in loss bit')
        # print(esp_lengths)
        
        target_esps = torch.split(
            labels[self.esp_column],
            esp_lengths
        )
           #  .flatten()     # torch.cat(
           
        # esp_length_column = labels[self.esp_length_column]
        
        # print('esp length column')
        # print(esp_length_column)
        
        #work out the inv distance chunks, these will be esp length * num_atoms
        inv_distance_chunks = [
            n_atoms * esp_len for n_atoms, esp_len in zip(n_atoms_per_mol, esp_lengths) 
        ]

        #split the inv distances by these chunks
        inv_distances = torch.split(
            labels[self.inv_distance_column],inv_distance_chunks
        )        
        
        charges = torch.split(prediction[self.charge_label].squeeze(), n_atoms_per_mol)
        
        entries_and_metrics = []
        for i, molecule in enumerate(molecules):
            error = self.evaluate_loss(
                molecules=molecule,
                labels={
                    self.inv_distance_column: inv_distances[i],
                    self.esp_column: target_esps[i],
                    self.esp_length_column : esp_lengths[i].view(-1).tolist(),
                },
                prediction={self.charge_label: charges[i]}
            )
            entries_and_metrics.append(
                (molecule, error * self.denominator / self.weight)
            )
        
        report_path = output_folder.joinpath(f"{self.esp_column}.html")
        create_molecule_label_report(
            entries_and_metrics=entries_and_metrics,
            metric_label=self.metric,
            output_path=report_path,
        )

        return report_path


LossCalculator = typing.Union[typing.Literal["ReadoutTarget", "DipoleTarget","ESPTarget"], str]


def get_loss_function(type_: LossCalculator) -> typing.Type[_BaseTarget]:
    """Get the loss calculator based on the type."""
    if type_.lower() == "readouttarget":
        return ReadoutTarget
    elif type_.lower() == "dipoletarget":
        return DipoleTarget
    elif type_.lower() == "esptarget":
        return ESPTarget
    else:
        raise NotImplementedError(f"Loss calculator {type_} not supported.")
