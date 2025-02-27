"""Models to define the train, val, test data sets."""
import typing

import pydantic

MetricType = typing.Literal["rmse", "mse", "mae"]


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class _BaseTarget:
    """Define a general base target config class to hold training/ evaluation data and settings."""

    metric: MetricType = pydantic.Field(
        ...,
        description="The metric to use when comparing the target data with the "
        "model output.",
    )
    denominator: float = pydantic.Field(
        1.0,
        description="The denominator which should be used to re-scale the metric value.",
    )
    weight: float = pydantic.Field(
        1.0, description="The weight that should be given to this metric."
    )


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class ReadoutTarget(_BaseTarget):
    """Defines a particular target to train / evaluate against."""

    column: str = pydantic.Field(
        ..., description="The column in the source field that contains the target data."
    )
    readout: str = pydantic.Field(
        ...,
        description="The name of the model readout that predicts the target data.",
    )


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class DipoleTarget(_BaseTarget):
    """Defines a Dipole specific target to train / evaluate against"""

    dipole_column: str = pydantic.Field(
        ...,
        description="The column in the source field that contains the dipole data in e*bohr",
    )
    conformation_column: str = pydantic.Field(
        ...,
        description="The column in the source field that contains the conformation the dipole should be evaluated at in **bohr**",
    )
    charge_label: str = pydantic.Field(
        ...,
        description="The name of the readout model that predicts the atomic charge to calculate the dipole.",
    )
    
@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class ESPTarget(_BaseTarget):
    """Defines a ESP specific target to train / evaluate against"""
    
    esp_column: str = pydantic.Field(
        ...,
        description="The column in the source field that contains the dipole data in Eh/e"
    ) 
    inv_distance_column: str  = pydantic.Field(
        ...,
        description="The column in the source field that contains the inverse distance between the conformer and the grid"
    )
    charge_label: str = pydantic.Field(
        ...,
        description="The name of the readout model that predicts the atomic charge to calculate the dipole.",
    )

    esp_length_column: str = pydantic.Field(
        ...,
        description="List of lengths the esps associated with the conformers"
    )
    
    ke: float = pydantic.Field(
        ...,                     
        description="The Coulomb constant, which should reflect the units used in the inv_distance",
    )
    
    #Optional[str] add multipoles here 

@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class Dataset:
    """Defines the targets to train / evaluate the model against during a given
    stage (i.e train, val, test)."""

    sources: typing.Optional[typing.List[str]] = pydantic.Field(
        None, description="The paths to the data."
    )
    targets: typing.List[typing.Union[ReadoutTarget, DipoleTarget, ESPTarget]] = pydantic.Field(
        ..., description="The targets to train / evaluate against."
    )
    batch_size: typing.Optional[int] = pydantic.Field(
        None, description="The batch size."
    )


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class DataConfig:
    """Defines the train, val, and test data sets."""

    training: Dataset = pydantic.Field(
        ...,
        description="The training data.",
    )
    validation: typing.Optional[Dataset] = pydantic.Field(
        None,
        description="The validation data.",
    )
    test: typing.Optional[Dataset] = pydantic.Field(
        None,
        description="The test data.",
    )
