## new data transition
import os
import time
import numpy as np
import sys
sys.path.append("/ssd/liwentao/ocp_main")
from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset, LmdbDataset
import ase.io
import ase.db
from ase import Atoms

from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
from dataloader import Dataloader
import random
random.seed(3407)

# dataset = LmdbDataset({"src": "s2ef/val/"})
# print(dataset[0])
# import pdb; pdb.set_trace()
split = "test"

# data_path = f"/ssd/liwentao/Transition1x/data/transition1x.h5"
# dataloaders = {
#     split: Dataloader(data_path, split, only_final=True),
# }


data_path = f"/ssd/liwentao/ocp_main/new_calculate/all_catalyst/new_{split}_fold1.h5"
dataloaders = {
    # "train": Dataloader(data_path, "train"),
    # "test": Dataloader(data_path, "test"),
    split: Dataloader(data_path),
    # "val": Dataloader(data_path, "val"),
}

raw_data = {}
energy_force_data = {}

a2g = AtomsToGraphs(
    max_neigh=20,
    radius=6,
    r_energy=False,
    r_forces=False,
    r_distances=False,
    r_edges=False,
    r_fixed=True,
    r_pbc=True,
)

rxn = []
for split, dataloader in dataloaders.items():
    # os.makedirs(f"catalyst_select_refine_shift/{split}", exist_ok=True)
    os.makedirs(f"../final_data/version2_cata_mask/{split}", exist_ok=True)
    # os.system(f"rm -rf transition1x_s2ef/{split}/*")

    db = lmdb.open(
        # f"catalyst_select_refine_shift/{split}/Rh.lmdb",
        f"../final_data/version2_cata_mask/{split}/Rh.lmdb",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    db_sync_freq = 1

    txn = db.begin(write=True)

    total_num = 0
    n_images = 9
    sampling_count = 0
    frame_idx = 0
    for idx,configuration in tqdm(enumerate(dataloader)):
        if configuration["rxn"] not in rxn:
            
            sampling_count = 0
            rxn.append(configuration["rxn"])
            frame_idx = 0
        
        
        # if split == "train" and sampling_count >= 50:
        #     continue
        # elif split != "train" and sampling_count < 50:
        #     sampling_count += 1
        #     continue
        # configuration = configuration['transition_state']
        atoms = Atoms(configuration["atomic_numbers"])
        atoms.set_positions(configuration["positions"])
        position_min = np.min(configuration["positions"])
        position_max = np.max(configuration["positions"])
        cell_constant = position_max - position_min + 50
        atoms.cell = [cell_constant, cell_constant, cell_constant]
        data_ef = {
            "energy": configuration["wB97x_6-31G(d).atomization_energy"],
            "forces": configuration["wB97x_6-31G(d).forces"],
        }

        data = a2g.convert(atoms)
        #assign sid
        data.sid = torch.LongTensor([0])

        data.force = torch.tensor(data_ef["forces"])
        data.y = torch.tensor(data_ef["energy"])
        data.pbc = torch.tensor([[True, True, True]])
        #assign fid
        data.fid = torch.LongTensor([int(total_num)])
        norm_force = data.force.norm(dim=-1)
        if split == "train":
            max_force = norm_force.max().item()
            if max_force < 0.05:
                continue
            # 创建力的掩码
            force_mask = torch.where(norm_force < 0.025, torch.rand(norm_force.shape) < 0.99, 0.0)
            force_mask_2 = torch.where(norm_force < 0.05, torch.rand(norm_force.shape) < 0.95, 0.0)
            # 取并
            force_mask_combined = torch.logical_or(force_mask, force_mask_2)
            data.fixed =force_mask_combined

        #assign tags, if available
        data.tags = torch.LongTensor(atoms.get_tags())
        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV and forces > |50| eV/A

        # no neighbor edge case check
        # import pdb; pdb.set_trace()

        txn.put(f"{total_num}".encode("ascii"), pickle.dumps(data, protocol=-1))

        # Sync and write every db_sync_freq steps
        if (total_num + 1) % db_sync_freq == 0:
            txn.commit()  # Commit changes
            txn = db.begin(write=True)  # Start a new transaction
        
        total_num += 1
        sampling_count += 1
        frame_idx += 1
    # Final commit for any remaining data not covered in the loop
    if (total_num + 1) % db_sync_freq != 0:
        txn.commit()
    print(f"Total number of configurations in {split} set: {total_num}")
    # txn = db.begin(write=True)
    # txn.put(f"length".encode("ascii"), pickle.dumps(total_num, protocol=-1))
    # txn.commit()
    db.sync()
    db.close()
    

# dataset = LmdbDataset({"src": f"catalyst_select_refine_shift/{split}"})
dataset = LmdbDataset({"src": f"../final_data/version2_cata_mask/{split}"})
print(dataset[0].y)
print(dataset[1].y)
# print(dataset[229].y)
# print(dataset[230].y)