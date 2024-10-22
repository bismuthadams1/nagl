import pathlib

import numpy
import pytest
from rdkit import Chem, Geometry

from nagl.features import AtomConnectivity, BondIsInRing
from nagl.molecules import DGLMolecule
from nagl.utilities.molecule import molecule_from_smiles


@pytest.fixture()
def rdkit_methane() -> Chem.Mol:
    molecule = molecule_from_smiles("C")
    conformer = Chem.Conformer(molecule.GetNumAtoms())

    coords = numpy.array(
        [
            [-0.0000658, -0.0000061, 0.0000215],
            [-0.0566733, 1.0873573, -0.0859463],
            [0.6194599, -0.3971111, -0.8071615],
            [-1.0042799, -0.4236047, -0.0695677],
            [0.4415590, -0.2666354, 0.9626540],
        ]
    )

    for i, coord in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(*coord))

    molecule.AddConformer(conformer)
    return molecule

@pytest.fixture()
def rdkit_nitrobromomolecule() -> Chem.Mol:
    molecule = molecule_from_smiles(
        '[H:13][C:1]12[C:7](=[N+:8]([C:9](=[N+:10]1[H:16])[Br:11])[H:15])[N:6]=[C:5]([N:4]([C:2]2=[O:3])[H:14])[N:12]([H:17])[H:18]'

    )
    conformer = Chem.Conformer(molecule.GetNumAtoms())

    coords = numpy.array([
       [ 1.63277354, -0.11236038, -0.39279205],
       [ 0.74095751, -2.53137717, -1.68588469],
       [ 1.73994915, -4.52075909, -1.53840216],
       [-1.42352152, -2.03949939, -3.19206292],
       [-2.89986577,  0.04989148, -2.9719376 ],
       [-2.62015597,  1.71281963, -0.97719408],
       [-0.61723862,  1.42917395,  0.31620954],
       [-0.13227705,  2.3559868 ,  2.70134792],
       [ 1.94491705,  1.19741057,  3.68079091],
       [ 2.95747561, -0.32566612,  2.00284822],
       [ 3.05617724,  1.72500005,  6.87518185],
       [-4.72632742,  0.49746631, -4.55198739],
       [ 2.79642457,  0.93891113, -1.75430589],
       [-1.96540194, -3.48995009, -4.32531903],
       [-1.29283246,  3.51698904,  3.68850491],
       [ 4.51816699, -1.3935594 ,  2.29418752],
       [-5.84043483,  2.03047394, -4.28319775],
       [-5.09614829, -0.61387828, -6.05968223]
    ])
    for i, coord in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(*coord))

    molecule.AddConformer(conformer)
    return molecule


@pytest.fixture()
def dgl_methane(rdkit_methane) -> DGLMolecule:
    return DGLMolecule.from_rdkit(
        rdkit_methane,
        [AtomConnectivity()],
        [BondIsInRing()],
    )


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path
