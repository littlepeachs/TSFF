# pylint: disable=stop-iteration-return

import h5py

# REFERENCE_ENERGIES = {
#     1: -13.62222753701504,
#     6: -1029.4130839658328,
#     7: -1484.8710358098756,
#     8: -2041.8396277138045,
#     9: -2712.8213146878606,
#     17:-12519.97763504221874,
# }

# REFERENCE_ENERGIES = {
#     1: -13.62222753701504,
#     6: -1029.4130839658328,
#     7: -1484.8710358098756,
#     8: -2041.8396277138045,
#     9: -2712.8213146878606,
#     15: -9280.27964225786878,
#     16:-10827.728979794708218,
#     17:-12519.97763504221874,
#     34:-65339.273477635335755,
#     35: -69780.071059292880568,
# }

# 1au等于27.2114 eV
REFERENCE_ENERGIES = {
    1: -18.53800713255103,
    6: -1033.8715833455858,
    7: -1488.9474440697306,
    8: -2045.7736943830027,
    9: -2716.254859006564,
    15: -9288.501881285394,
    17: -12522.30544826637,
    45: -3011.80735487193,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp):
    """ Iterates through a h5 group """

    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)

    for energy, force, positions in zip(energies, forces, positions):
        d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }

        yield d


class Dataloader:
    """
    Can iterate through h5 data set for paper ####

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    """

    def __init__(self, hdf5_file, datasplit="data", only_final=False):
        self.hdf5_file = hdf5_file
        self.only_final = only_final

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    reactant = next(generator(formula, rxn, subgrp["reactant"]))
                    product = next(generator(formula, rxn, subgrp["product"]))

                    if self.only_final:
                        transition_state = next(
                            generator(formula, rxn, subgrp["transition_state"])
                        )
                        yield {
                            "rxn": rxn,
                            "reactant": reactant,
                            "product": product,
                            "transition_state": transition_state,
                        }
                    else:
                        yield reactant
                        yield product
                        for molecule in generator(formula, rxn, subgrp):
                            yield molecule

class OPTDataloader:
    """
    Can iterate through h5 data set for paper ####

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    """

    def __init__(self, hdf5_file, datasplit="data", only_final=False):
        self.hdf5_file = hdf5_file
        self.only_final = only_final

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    for molecule in generator(formula, rxn, subgrp):
                        yield molecule