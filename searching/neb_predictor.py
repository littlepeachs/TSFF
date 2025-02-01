from ase.io.trajectory import Trajectory

import logging
import os
import pickle
# from _typeshed import SupportsDunderLT, SupportsDunderGT
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
from .interpolate import interpolate_custom,rotate_reactant,interpolate_custom_idpp
from ase.optimize.bfgs import BFGS

model_type = "Eq"
# interpolate = "custom_from_scratch"
interpolate = "t1x"

class OCPCalculatorMod(OCPCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, config_yml: Optional[str] = None, checkpoint_path: Optional[str] = None,
                 trainer: Optional[str] = None, cutoff: int = 6, max_neighbors: int = 20, cpu: bool = False,
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
        self.trainer.config['cmd']['seed'] = 0
        self.trainer.set_seed(0)


class MLFF_NEB_Predictor():
    def __init__(self,
                 args,
                 neb_config_path: str,
                 cineb_config_path: str,
                 checkpoint_path: str = None,
                 save_path: str = None,
                 save_lmdb_size=1099511627776 * 2,
                 dataset_pth: str = None,
                 device_id=0,  # [0,1] if use gpu 0 and 1
                 cpu: bool = False,
                 total_chunk: int = 1,
                 chunk: int = 0,
                 ) -> None:
        """
        Parameters:
            neb_config_path: str, the path of the neb configuration file
            vib_config_path: str, the path of the vib configuration file
            checkpoint_path: str, the path of the checkpoint
            save_path: str, the path to save the results
            save_lmdb_size: int, the size of the lmdb
            dataset_pth: str, the path of the dataset
        """
        self.opt_model_path = args.opt_model_path
        self.neb_params = yaml.safe_load(open(neb_config_path))
        self.cineb_params = yaml.safe_load(open(cineb_config_path))
        self.save_path = save_path
        self.dataset_pth = dataset_pth

        self.device_id = device_id
        self.cpu = cpu
        self.total_chunk = total_chunk
        self.chunk = chunk

        self.init_dataset_calc(checkpoint_path=checkpoint_path, save_lmdb_size=save_lmdb_size)

    def init_dataset_calc(self, save_lmdb_size, checkpoint_path: str = None):

        self.data_lmdb = lmdb.open(self.dataset_pth, subdir=False, meminit=False, map_async=True, readonly=True)
        self.data_length = pickle.loads(self.data_lmdb.begin(write=False).get("length".encode("ascii")))

        self.save_lmdb = lmdb.open(os.path.join(self.save_path, f"save_{self.chunk}.lmdb"),
                                   map_size=int(save_lmdb_size), subdir=False,
                                   meminit=False,
                                   map_async=True)
        self.calc = OCPCalculatorMod(checkpoint_path=checkpoint_path, cpu=self.cpu, device_id=self.device_id)
        if self.opt_model_path is not None:
            self.opt_calc = OCPCalculatorMod(checkpoint_path=self.opt_model_path, cpu=self.cpu, device_id=self.device_id)


    # todo: create images between the reactant and product
    def data_process(self,
                     data,
                     n_images: int,
                     ) -> List[Atoms]:
        """
        Parameters:
            data: the data for the NEB calculation
            n_images: int, the number of images between the reactant and product

        1. convert the reactant and product to ase object
        2. create images between the reactant and product

        Return:
            List[Atoms]: initial images
        """

        reactant = data.reactant
        product = data.product

        reactant = torch2ase(reactant)
        product = torch2ase(product)

        images = [reactant]
        images += [reactant.copy() for _ in range(n_images)] 
        images += [product]
        return images

    def run_neb(self,
                images: List[Atoms],
                reaction_id: str = None, ) -> List[Atoms]:
        """
        Parameters:
            images: List[Atoms], the images for the NEB calculation
            reaction_id: str, the reaction id
            device: int, the device id

        1. predict the energy and forces for the images with ml model
        2. search the reaction path with NEB algorithm

        Return:
            List[Atoms]: the optimized images
        """

        normal_terminate = True
        if self.opt_model_path is not None:
            images[0].set_calculator(self.opt_calc)
            images[-1].set_calculator(self.opt_calc)
            reactant_opt = BFGS(images[0],logfile=None)
            converge_reactant = reactant_opt.run(fmax=0.05, steps=200)
            product_opt = BFGS(images[-1],logfile=None)
            converge_product = product_opt.run(fmax=0.05, steps=200)
            if converge_reactant and converge_product:
                print("成功优化结构",flush=True)
            else:
                print("优化结构失败",flush=True)


        if self.neb_params["neb_algorithm"] == "NEB":
            neb = NEB(images, climb=self.neb_params["climb"], method=self.neb_params["method"],
                    allow_shared_calculator=True)
            # neb = NEB(images,allow_shared_calculator=True)
            if interpolate == "custom":
                images = interpolate_custom(images[0],images[-1],num_intermediate=self.neb_params["n_images"])
                neb= NEB(images, climb=self.neb_params["climb"], method=self.neb_params["method"],
                    allow_shared_calculator=True)
                write(f"{self.save_path}/neb_init/init_{reaction_id}.traj",images)
            elif interpolate == "custom_from_scratch":
                reaction_name = reaction_id[3:]
                reactant_path = f"/ssd/liwentao/ocp_main/new_calculate/all_catalyst/{reaction_name}/{reaction_name}_reactant.xyz"
                product_path = f"/ssd/liwentao/ocp_main/new_calculate/all_catalyst/{reaction_name}/{reaction_name}_product.xyz"
                opt_reactant = images[0].copy()
                opt_product = images[-1].copy()
                images[0] = read(reactant_path)
                images[-1] = read(product_path)
                images = interpolate_custom(images[0],images[-1],num_intermediate=self.neb_params["n_images"])
                images[0] = opt_reactant
                images[-1] = opt_product
                neb= NEB(images, climb=self.neb_params["climb"], method=self.neb_params["method"],
                    allow_shared_calculator=True)
                write(f"{self.save_path}/neb_init/init_{reaction_id}.traj",images)
            else:
                neb.interpolate(method=self.neb_params["interpolation"])

                traj_filename = f'{self.save_path}/neb_init/init_{reaction_id}.traj'  # 轨迹文件名
                traj = Trajectory(traj_filename, 'w')  # 打开文件用于写入
                # 遍历所有的图像并将它们添加到轨迹文件中
                for image in neb.images:
                    traj.write(image)
                traj.close()

            for image in images:
                image.set_calculator(self.calc)

            neb_traj_save_path = f"{self.save_path}/neb/neb_{reaction_id}.traj"
            cineb_traj_save_path = f"{self.save_path}/neb/cineb_{reaction_id}.traj"
            if self.neb_params["optimizer"] == "FIRE":
                optimizer = FIRE(neb, trajectory=neb_traj_save_path,maxstep=self.neb_params["max_step"])
            elif self.neb_params["optimizer"] == "LBFGS":
                optimizer = LBFGS(neb, trajectory=neb_traj_save_path,maxstep=self.neb_params["neb_max_step"])
            elif self.neb_params["optimizer"] == "BFGS":
                optimizer = BFGS(neb, trajectory=neb_traj_save_path,maxstep=self.neb_params["neb_max_step"])
            elif self.neb_params["optimizer"] == "ODE":
                optimizer = NEBOptimizer(neb,trajectory=None,method='ODE',hmin=0.0005)
            else:
                raise NotImplementedError(f"Optimizer {self.neb_params['optimizer']} is not implemented")

            # result = True means the optimization is converged
            try:
                converge_result = optimizer.run(fmax=self.neb_params['fmax'], steps=self.neb_params['steps'])
            except AssertionError as e:
                print(f"发生了断言错误：{e}")
                converge_result = False
                normal_terminate=False
            print("NEB optimization is done",flush=True)
        


        if converge_result and self.neb_params["restart"]:
            restart_neb = NEB(images, climb=self.cineb_params["climb"], method=self.cineb_params["method"],
                                 allow_shared_calculator=True)
            # restart_neb = NEB(images, climb=self.neb_params["climb"],allow_shared_calculator=True)
            if self.neb_params["optimizer"] == "BFGS":
                optimizer = BFGS(restart_neb, trajectory=cineb_traj_save_path,maxstep=self.cineb_params["cineb_max_step"])
            elif self.neb_params["optimizer"] == "ODE":
                optimizer = NEBOptimizer(restart_neb,trajectory=None ,method='ODE',hmin=0.0005)

            try:
                converge_result = optimizer.run(fmax=self.cineb_params['fmax'], steps=self.cineb_params['steps'])
            except Exception as e:
                print(f"发生了断言错误：{e}")
                converge_result = False
                normal_terminate = False
                
            print("CINEB optimization is done",flush=True)
        if not converge_result:
            print("Not converged",flush=True)
        
        self.plot_neb_mep(images,reaction_id)
        return images, converge_result,normal_terminate

    def plot_neb_mep(self,
                     images: List[Atoms],
                     reaction_id: str = None,):
        neb_tools = NEBTools(images)
        fit = neb_tools.get_fit()

        fig, ax = plt.subplots()
        ax.plot(
            fit.fit_path, fit.fit_energies, label=f"Barrier: {max(fit.fit_energies):.2f} eV"
        )

        ax.patch.set_facecolor("#E8E8E8")
        ax.grid(color="w")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_ylabel("Energy [eV]")
        ax.set_xlabel("Reaction Coordinate [AA]")
        ax.legend()
        fig.savefig(os.path.join(self.save_path,"mep_fig",f"neb_{reaction_id}.png"))


    def plot_energy(self,
                    images: List[Atoms],
                    reaction_id: str = None,
                    save_plot: bool = False):
        """
        Parameters:
            images: List[Atoms], the images for the NEB calculation
            reaction_id: str, the reaction id
            save_plot: bool, whether to save the energy plot

        Return:
            Atoms: the predicted transition state
        """

        energies = [image.get_potential_energy() for image in images]

        # 绘制能量变化折线图
        energies_kcal = [(i - energies[0]) * mol / kcal for i in energies]  # 将能量单位从ew转化为kcal/mol，并且以第一个结构为0点
        predict_ts_energy = max(energies)
        predict_ts = images[energies.index(max(energies))]
        write(f"{self.save_path}/neb_ci/{reaction_id}_predcit_ts.xyz", predict_ts)

        print(f"反应的预测能垒：{round((max(energies) - energies[0]), 4)} (eV)")
        print(f"反应物的能量：{round(energies[0], 4)} (eV)")
        print(f"过渡态的能量：{round(max(energies), 4)} (eV)")

        if save_plot:
            neb_tools = NEBTools(images)
            fit = neb_tools.get_fit()

            fig, ax = plt.subplots()
            ax.plot(
                fit.fit_path, fit.fit_energies, label=f"Barrier: {max(fit.fit_energies):.2f} eV"
            )

            ax.patch.set_facecolor("#E8E8E8")
            ax.grid(color="w")
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.set_ylabel("Energy [eV]")
            ax.set_xlabel("Reaction Coordinate [AA]")
            ax.legend()
            fig.savefig(os.path.join("mep_fig", f"cineb_{reaction_id}_9.png"))

        return predict_ts, predict_ts_energy
        # write(neb_path, images)        

    def predict(self, i):
        """
        Parameters:
            data: the data for the NEB calculation

        Return:
            Atoms: the predicted transition state
        """
        data = pickle.loads(self.data_lmdb.begin(write=False).get(f"{i}".encode("ascii")))
        images = self.data_process(data, self.neb_params['n_images'])
        reaction_id = data.reaction_id

        if reaction_id == "rxnCat274":
            return data,None,False,None,None
        
        print("The reaction is:", reaction_id,flush=True)

        images, converge_result,normal_terminate = self.run_neb(images, reaction_id=reaction_id)
        if not normal_terminate:
            return data, None, False, None, None
        
        predict_ts, predict_ts_energy = self.plot_energy(images, reaction_id=reaction_id)

        return data, predict_ts, converge_result, predict_ts_energy

        # todo merge the main function into MLFF_NEB_predictor()

    def main(self, sync_freq: int = 1):
        save_txn = self.save_lmdb.begin(write=True)
        save_n = 0
        converge =0
        print("NEB Config:", self.neb_params)
        print("CINEB Config:", self.cineb_params)

        for i in tqdm(range(self.data_length)[self.chunk::self.total_chunk]):
            data, predict_ts, converge_result, predict_ts_energy = self.predict(i)
            if predict_ts is None:
                print(f"AssertionError in {data.reaction_id}, atom too close",flush=True)
                continue
            save_data = NEBData()
            save_data.reaction_id = data.reaction_id
            save_data.predict_ts_structure = predict_ts
            save_data.converge_result = converge_result
            save_data.reactant_structure = torch2ase(data.reactant)
            save_data.product_structure = torch2ase(data.product)

            if hasattr(data, "ts"):
                save_data.gt_ts_structure = torch2ase(data.ts)
                save_data.gt_ts_structure.arrays["energy"] = np.array([data.ts.y])


            save_data.predict_ts_structure.arrays["energy"] = np.array([predict_ts_energy])
            save_data.predict_ts_structure._calc=None
            save_data.neb_settings = self.neb_params
            save_data.save_pth = self.save_path
            
            save_txn.put(f"{save_n}".encode("ascii"), pickle.dumps(save_data))
            save_n += 1

            if save_n % sync_freq == 0:
                save_txn.commit()
                self.save_lmdb.sync()
                save_txn = self.save_lmdb.begin(write=True)

        save_txn.put("length".encode("ascii"), pickle.dumps(save_n))
        save_txn.commit()
        self.save_lmdb.sync()
        self.save_lmdb.close()
        



class NEBData:
    def __init__(self):
        self.reaction_id = None
        self.surface_id = None
        self.site_id = None
        self.reactant_structure = None
        self.product_structure = None
        self.predict_ts_structure = None
        self.predict_ts_zero_point_energy = None
        self.predict_ts_entropy = None
        self.predict_ts_free_energy = None
        self.predict_ts_frequency = None
        self.gt_ts_structure = None
        self.neb_traj = None
        self.neb_settings = None
        self.converge_result = None

    def __repr__(self):
        return f"NEBData: {self.reaction_id}, ts_energy: {self.predict_ts_structure.get_potential_energy()}, " \
               f"converge: {self.converge_result}"


def torch2ase(data):
    react_atomic_numbers = data["atomic_numbers"].to(torch.int64).tolist()
    ase_obj = Atoms(numbers=react_atomic_numbers, positions=data.pos.numpy())
    return ase_obj
