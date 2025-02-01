import matplotlib.pyplot as plt
import pickle
import lmdb
import torch
import sys
import numpy as np
sys.path.append('../')
from data_processing.reaction_data_creation import reaction_data_object
from ase.io import read,write
from torch.nn.functional import mse_loss
import os
import argparse

args = argparse.ArgumentParser()
args.add_argument("--file_name", type=str, default="./save/t1x_neb/save_0.lmdb" )
args.add_argument("--label",type=str,default="t1x_neb")
args.add_argument("--data_length",type=int,default=10)
args = args.parse_args()

predict_energy = []
gt_energy = []
predict_energy_converge = []
gt_energy_converge = []
not_converge_count =[]
error_reaction = []
error_reaction_energy = []
reaction_ids = []


file_name = args.file_name
label = args.label
folder_path = os.path.dirname(file_name)
data_lmdb = lmdb.open(file_name, subdir=False,meminit=False,map_async=True,readonly=True)
# data_length = pickle.loads(data_lmdb.begin(write=False).get("length".encode("ascii")))
data_length=args.data_length

for i in range(data_length):
    data = pickle.loads(data_lmdb.begin(write=False).get(f"{i}".encode("ascii")))
    # print("reaction_id:",data.reaction_id) 
    # 在cal_barrier中查找当前reaction_id的索引

    # write(f"./data/save/neb_ci/predict_ts_{react_data.reactant.rxn}.xyz",data.predict_ts_structure)
    
    write(f"{folder_path}/neb_ci/gt_ts_{data.reaction_id}.xyz",data.gt_ts_structure)
    
    os.system(f"calculate_rmsd {folder_path}/neb_ci/{data.reaction_id}_predcit_ts.xyz {folder_path}/neb_ci/gt_ts_{data.reaction_id}.xyz >> ./experiment_logs/rmsd_{label}.log")
    reaction_ids.append(data.reaction_id)
    
    predict_energy.append(data.predict_ts_structure.arrays["energy"][0]-4.243785228188971*data.predict_ts_structure.positions.shape[0]) # -4.243785228188971*data.predict_ts_structure.positions.shape[0]
    
    # predict_energy.append(data.predict_ts_structure.arrays["energy"][0])
    gt_energy.append(data.gt_ts_structure.arrays["energy"][0])
    if data.converge_result:
        predict_energy_converge.append(data.predict_ts_structure.arrays["energy"][0])
        gt_energy_converge.append(data.gt_ts_structure.arrays["energy"][0])
    if not data.converge_result:
        not_converge_count.append(i)
        
        energy_mae = np.abs(data.predict_ts_structure.arrays["energy"][0] -data.gt_ts_structure.arrays["energy"][0]-4.243785228188971*data.predict_ts_structure.positions.shape[0]) 
       
        print("energy mae:",energy_mae)

        error_reaction.append(data.reaction_id)
        error_reaction_energy.append(energy_mae)

mae_array = np.abs(np.array(predict_energy)-np.array(gt_energy))

mae_dict = {reaction_id:mae for reaction_id,mae in zip(reaction_ids,mae_array)}

print("MAE mean:",np.mean(mae_array))
print("MAE median:",np.median(mae_array))

rmsd = []
with open(f"./experiment_logs/rmsd_{label}.log","r") as f:
    for line in f:
        rmsd.append(float(line))
rmsd = torch.tensor(rmsd)

print("RMSD mean:",torch.mean(rmsd))
print("RMSD median:",torch.median(rmsd))