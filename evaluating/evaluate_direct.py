from ase.io.trajectory import Trajectory
# import nni
import logging
import os
from data_processing.reaction_data_creation import reaction_data_object
import pickle
from ocpmodels.datasets import LmdbDataset
from ocpmodels.common.data_parallel import BalancedBatchSampler, OCPCollater
from ocpmodels.common.registry import registry
from ocpmodels.common import gp_utils,distutils
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import torch
import yaml
from ase.io import read, write
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ase.neb import NEBTools
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from ase import Atoms
from ase.neb import NEB
from ase.mep.neb import NEBOptimizer
from ase.optimize import LBFGS, FIRE, BFGS,ODE12r
import matplotlib.pyplot as plt
# 导入kcal和mol单位，用于单位转换
from ase.units import kcal, mol
from ase.mep.neb import NEBTools
import numpy as np
import sys
from typing import List, Union, Tuple, Any, Optional
from ase.optimize.precon import Exp_FF
from tqdm import tqdm

import lmdb

from ase.optimize.bfgs import BFGS
from pyscf import gto, dft
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Evaluate direct")
parser.add_argument("--run_num", type=int, default=10, help="run number")
args = parser.parse_args()


def calculate_rmsd(coords1, coords2):
    """
    计算两个坐标矩阵之间的RMSD值,使用Kabsch算法进行结构对齐
    
    参数:
        coords1 (numpy.ndarray): 第一组原子坐标
        coords2 (numpy.ndarray): 第二组原子坐标
    
    返回:
        float: RMSD值
    """
    assert coords1.shape == coords2.shape, "坐标矩阵的形状必须相同"
    
    # 使用Kabsch算法对齐结构
    # 中心化坐标
    P_centered = coords1 - np.mean(coords1, axis=0)
    Q_centered = coords2 - np.mean(coords2, axis=0)

    # 计算协方差矩阵
    C = np.dot(Q_centered.T, P_centered)

    # 使用奇异值分解(SVD)计算最优旋转矩阵
    V, S, Wt = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # 计算旋转矩阵U
    U = np.dot(V, Wt)
    
    # 旋转coords2以对齐coords1
    coords2_aligned = np.dot(Q_centered, U) + np.mean(coords1, axis=0)
    
    # 计算RMSD
    diff = coords1 - coords2_aligned
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

class OCPCalculatorMod(OCPCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, config_yml: Optional[str] = None, checkpoint_path: Optional[str] = None,
                 trainer: Optional[str] = None, cutoff: int = 12, max_neighbors: int = 20, cpu: bool = False,
                 device_id: int = 0) -> None:
        """
        OCP-ASE Calculator

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint_path (str):
                Path to trained checkpoint.
            trainer (str):
                OCP trainer to be used. "forces" for S2EF, "energy" for IS2RE.
            cutoff (int):
                Cutoff radius to be used for data preprocessing.
            max_neighbors (int):
                Maximum amount of neighbors to store for a given atom.
            cpu (bool):
                Whether to load and run the model on CPU. Set `False` for GPU.
        """
        super().__init__(config_yml, checkpoint_path, trainer, cutoff, max_neighbors,seed=0)

        from ocpmodels.common.registry import registry

        config = self.config
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu")
        )
        if "normalizer" not in config:
            config["dataset"]["src"] = None
        #     config["normalizer"] = config["dataset"]
        self.trainer = registry.get_trainer_class(
            config.get("trainer", "energy")
        )(
            task=config["task"],
            model=config["model"],
            dataset=[config["dataset"]],
            # normalizer=config["normalizer"],
            outputs=config["outputs"],
            loss_fns=config["loss_fns"],
            eval_metrics=config["eval_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=device_id,
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=False,
        )
        if checkpoint_path is not None:
            self.load_checkpoint(
                checkpoint_path=checkpoint_path, checkpoint=checkpoint
            )
        self.trainer.set_seed(0)
        self.trainer.config['cmd']['seed'] = 0

def get_sampler(dataset, batch_size: int, shuffle: bool) -> BalancedBatchSampler:
    balancing_mode = "atoms"
    force_balancing = False

    if gp_utils.initialized():
        num_replicas = gp_utils.get_dp_world_size()
        rank = gp_utils.get_dp_rank()
    else:
        num_replicas = distutils.get_world_size()
        rank = distutils.get_rank()
    sampler = BalancedBatchSampler(
        dataset,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=rank,
        device=1,
        mode=balancing_mode,
        shuffle=shuffle,
        force_balancing=force_balancing,
    )
    return sampler

def get_dataloader(dataset, sampler, ocp_collater) -> DataLoader:
    loader = DataLoader(
        dataset,
        collate_fn=ocp_collater,
        num_workers=8,
        pin_memory=True,
        batch_sampler=sampler,
    )
    return loader

def load_datasets(train_dataset) -> None:
    ocp_collater = OCPCollater(
        otf_graph=True
    )
    train_loader = None

    train_sampler = get_sampler(
        train_dataset,
        1,
        shuffle=False,
    )
    train_loader = get_dataloader(
        train_dataset,
        train_sampler,
        ocp_collater
    )
    return train_loader

calc = OCPCalculatorMod(checkpoint_path=f"./final_checkpoints/cata_ckpt/best_checkpoint.pt",cpu=False, device_id=0)

# data_lmdb = lmdb.open("/ssd/liwentao/MLFF-TS-Search/data/catalyst_save/pretrain_select_10/save_0.lmdb", subdir=False, meminit=False, map_async=True, readonly=True)
ori_lmdb = lmdb.open(f"./data/cata_data/reaction/reaction.lmdb", subdir=False, meminit=False, map_async=True, readonly=True)
# result_lmdb = lmdb.open("/ssd/liwentao/MLFF-TS-Search/data/catalyst_save/all_cata_np_40_03_mask/save_0.lmdb", subdir=False, meminit=False, map_async=True, readonly=True)
result_lmdb = lmdb.open(f"./save/cata_neb/save_0.lmdb", subdir=False, meminit=False, map_async=True, readonly=True)
folder_path = f"./save/cata_neb"

mae_list = []
ts_mae_list = []
ts_mae_dict = {}
ts_energy_dict = {}
gt_energy_dict = {}
rmsd_list = []
rmsd_dict = {}

predict_energy_list = []
gt_energy_list = []

sample_force_list = {}
sample_energy_list = {}

converged_energy_list = []
gt_converged_energy_list = []
converged_rmsd_list = []

for i in range(args.run_num):
    ori_data = pickle.loads(ori_lmdb.begin(write=False).get(f"{i}".encode("ascii")))
    # import pdb;pdb.set_trace()
    result_data = pickle.loads(result_lmdb.begin(write=False).get(f"{i}".encode("ascii")))

    reaction_id = result_data.reaction_id
    
    
    # import pdb;pdb.set_trace()
    ori_data.reactant.cell = torch.Tensor([[[50.0,0.0,0.0],[0.0,50.0,0.0],[0.0,0.0,50.0]]])
    data_list = [ori_data.reactant]
    
    data = OCPCollater(otf_graph=True)(data_list)
    data = data.to(torch.device("cuda:0"))
    outputs = calc.trainer.model.forward(data)
    reactant_energy = outputs['energy'].detach().cpu().numpy()    

    ts_energy = result_data.predict_ts_structure.arrays["energy"][0]

    predict_barrier = ts_energy - reactant_energy
    gt_barrier = ori_data.ts.y - ori_data.reactant.y
    

    if result_data.converge_result is True:
        converged_energy_list.append(ts_energy)
        gt_converged_energy_list.append(ori_data.ts.y.item())
        ts_rmsd = calculate_rmsd(result_data.predict_ts_structure.arrays["positions"],result_data.gt_ts_structure.arrays["positions"])
        # print(f"ts_rmsd:{ts_rmsd}")
        converged_rmsd_list.append(ts_rmsd)
    # predict_energy_list.append(predict_barrier[0])
    # gt_energy_list.append(gt_barrier.item())
    mae = np.abs(predict_barrier[0] - gt_barrier.item())
    # print("mae:",mae)
    # ts_mae = np.abs(predict_ts_energy[0] - ori_data.ts.y.item())
    ts_mae = np.abs(result_data.predict_ts_structure.arrays['energy'][0]-ori_data.ts.y.item())

    write(f"{folder_path}/neb_ci/gt_ts_{reaction_id}.xyz",result_data.gt_ts_structure)
    write(f"{folder_path}/neb_ci/predict_ts_{reaction_id}.xyz",result_data.predict_ts_structure)
    
    
    predict_energy_list.append(ts_energy)
    gt_energy_list.append(ori_data.ts.y.item())
    mae_list.append(mae)
    
    ts_mae_list.append(ts_mae)
    ts_mae_dict[reaction_id] = ts_mae
    ts_energy_dict[reaction_id] = result_data.predict_ts_structure.arrays['energy'][0]
    gt_energy_dict[reaction_id] = ori_data.ts.y.item()
    
    ts_rmsd = calculate_rmsd(result_data.predict_ts_structure.arrays["positions"],result_data.gt_ts_structure.arrays["positions"])
    # print(f"ts_rmsd:{ts_rmsd}")
    rmsd_list.append(ts_rmsd)
    rmsd_dict[reaction_id] = ts_rmsd

# print("mae_list:",mae_list)
print("能垒的mae_mean:",np.mean(mae_list))
print("能垒的mae_std:",np.std(mae_list))
mae_array = np.array(mae_list)
percentiles = [25, 50, 75]
quantiles = np.percentile(mae_array, percentiles)

print(f"25% 分位数: {quantiles[0]:.4f}")
print(f"50% 分位数 (中位数): {quantiles[1]:.4f}") 
print(f"75% 分位数: {quantiles[2]:.4f}")

ts_mae_array = np.array(ts_mae_list)

print("ts_mae_mean:",np.mean(ts_mae_array))
print("ts_mae_std:",np.std(ts_mae_array))
ts_mae_array = np.array(ts_mae_array)
quantiles = np.percentile(ts_mae_array, percentiles)
print(f"25% 分位数: {quantiles[0]:.4f}")
print(f"50% 分位数 (中位数): {quantiles[1]:.4f}") 
print(f"75% 分位数: {quantiles[2]:.4f}")

print("ts_rmsd_mean:",np.mean(rmsd_list))   
