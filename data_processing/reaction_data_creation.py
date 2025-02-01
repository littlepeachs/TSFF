import torch
from argparse import ArgumentParser
import os
import lmdb
from ase import Atoms
from ase.io import write
import sys
sys.path.append('../')
import numpy as np
from .dataloader import Dataloader
import pickle
import tqdm
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset,LmdbDataset # type: ignore
from ocpmodels.preprocessing import AtomsToGraphs

a2g = AtomsToGraphs(
    max_neigh=20,
    radius=6,
    r_energy=False,    
    r_forces=False,   
    r_distances=False,
    r_fixed=True,
)

def get_cell(configuration):
    position_min = np.min(configuration.positions)
    position_max = np.max(configuration.positions)
    cell_constant = position_max - position_min+50
    configuration.cell = [cell_constant,cell_constant,cell_constant]
    return configuration

split = "test"
data_path=f"/ssd/liwentao/ocp_main/new_calculate/all_gen_neb/all_gen_neb_test.h5"
# data_path=f"ood_data_{split}.h5"
dataloaders = {
        # "val": Dataloader(data_path, "val", only_final=True),
        # "test": Dataloader(data_path, "test", only_final=True),
        # split: Dataloader(data_path, only_final=True),
        "test": Dataloader(data_path, only_final=True),
        # "train": Dataloader("/ssd/liwentao/MLFF-TS-Search/final_data/all_catalyst_fold1/Rh_train_fold1.h5", only_final=True),
    }
class reaction_data_object:
    def __init__(self) -> None:
        self.reactant = None
        self.product = None
        self.ts = None
        self.reaction_id = None

def main(args):  # pylint: disable=redefined-outer-name
    react_split_data = []
    react_split_ef_data=[]
    product_split_data = []
    product_split_ef_data=[]
    ts_split_data = []
    ts_split_ef_data=[]
    total_num = 0
    os.makedirs(f"../final_data/generate_data_reaction_more/", exist_ok=True)
    db = lmdb.open(
        f"../final_data/generate_data_reaction_more/reaction.lmdb",
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    for split, dataloader in dataloaders.items():
        # error_reaction = ['rxnCat398',"rxnCat770","Cat809"] #'rxn10005', 'rxn8550', 'rxn3684', 'rxn4441', 'rxn0249', 'rxn1064', 'rxn1089', 'rxn1277', 'rxn1282', 'rxn1284', 'rxn1286', 'rxn1786'
        
        for idx, configurations in enumerate(dataloader):
            print('rxn:',configurations['rxn'])
            rxn = configurations['rxn']
            reactant = configurations['reactant']
            product = configurations['product']
            transition_state = configurations['transition_state']

            # os.makedirs(args.output, exist_ok=True)
            # if rxn in error_reaction:
            reactant_atoms = Atoms(positions=reactant['positions'], numbers = reactant['atomic_numbers'])
            reactant_atoms = get_cell(reactant_atoms)
            react_split_data.append(reactant_atoms)
            react_split_ef_data.append({
                "energy": reactant["wB97x_6-31G(d).atomization_energy"],
                "forces": reactant["wB97x_6-31G(d).forces"],
                "rxn": rxn
            })

            product_atoms = Atoms(positions=product['positions'], numbers = product['atomic_numbers'])
            product_atoms = get_cell(product_atoms)
            product_split_data.append(product_atoms)
            product_split_ef_data.append({
                "energy": product["wB97x_6-31G(d).atomization_energy"],
                "forces": product["wB97x_6-31G(d).forces"],
                "rxn": rxn

            })
            
            transition_state_atoms = Atoms(positions=transition_state['positions'], numbers = transition_state['atomic_numbers'])
            transition_state_atoms = get_cell(transition_state_atoms)
            ts_split_data.append(transition_state_atoms)
            ts_split_ef_data.append({
                "energy": transition_state["wB97x_6-31G(d).atomization_energy"],
                "forces": transition_state["wB97x_6-31G(d).forces"],
                "rxn": rxn
            })
            
            # writing_boundary=1000
            
            # # import pdb;pdb.set_trace()
            # if len(react_split_data)==writing_boundary:   # len(react_split_data)==writing_boundary and 

    total_num+=len(react_split_data)
    print('total_num:',total_num)

    react_data_objects = a2g.convert_all(react_split_data, disable_tqdm=True)
    prod_data_objects = a2g.convert_all(product_split_data, disable_tqdm=True)
    ts_data_objects = a2g.convert_all(ts_split_data, disable_tqdm=True)
    for fid, (react_data,prod_data,ts_data) in enumerate(zip(react_data_objects,prod_data_objects,ts_data_objects)):
        data = reaction_data_object()
        react_data.sid = torch.LongTensor([0])
        react_data.force = torch.tensor(react_split_ef_data[fid]["forces"])
        react_data.y = torch.tensor(react_split_ef_data[fid]["energy"])
        #assign fid
        react_data.fid = torch.LongTensor([int(fid)])
        react_data.rxn = react_split_ef_data[fid]["rxn"]
        #assign tags, if available
        react_data.tags = torch.LongTensor(react_split_data[fid].get_tags())
        react_data.cell = torch.tensor([50.0,50.0,50.0])
        react_data.pbc = torch.tensor([[True, True, True]])

        prod_data.sid = torch.LongTensor([0])
        prod_data.force = torch.tensor(product_split_ef_data[fid]["forces"])
        prod_data.y = torch.tensor(product_split_ef_data[fid]["energy"])
        #assign fid
        prod_data.fid = torch.LongTensor([int(fid)])
        prod_data.rxn = product_split_ef_data[fid]["rxn"]
        #assign tags, if available
        prod_data.tags = torch.LongTensor(product_split_data[fid].get_tags())
        prod_data.cell = torch.tensor([50.0,50.0,50.0])
        prod_data.pbc = torch.tensor([[True, True, True]])

        ts_data.sid = torch.LongTensor([0])
        ts_data.force = torch.tensor(ts_split_ef_data[fid]["forces"])
        ts_data.y = torch.tensor(ts_split_ef_data[fid]["energy"])
        #assign fid
        ts_data.fid = torch.LongTensor([int(fid)])
        ts_data.rxn = ts_split_ef_data[fid]["rxn"]
        #assign tags, if available
        ts_data.tags = torch.LongTensor(ts_split_data[fid].get_tags())
        ts_data.cell = torch.tensor([50.0,50.0,50.0])
        ts_data.pbc = torch.tensor([[True, True, True]])

        data.reactant = react_data
        data.product = prod_data
        data.ts = ts_data
        data.reaction_id = product_split_ef_data[fid]["rxn"]

        txn = db.begin(write=True)
        txn.put(f"{fid}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
    txn = db.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(total_num, protocol=-1))
    txn.commit()
    db.sync()
    db.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    # os.system("rm -rf ./reaction_data/*")
    # parser.add_argument("h5file", nargs='?',type=str,default="/home/liwentao/Transition1x/data/transition1x.h55")
    # parser.add_argument("output", nargs='?',type=str,default="./reaction_data_test/")
    args = parser.parse_args()
    main(args)
    dataset = LmdbDataset({"src": f"../final_data/generate_data_reaction_more/"})
    print(len(dataset))
    print(dataset[0])
    # print(dataset[1])

