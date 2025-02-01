import argparse
import json
import os
import multiprocessing
import matplotlib.pyplot as plt
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
import ase.db
from ase.calculators.orca import ORCA
from ase.io import read, write
from ase.neb import NEB, NEBTools
from ase.mep.neb import NEBOptimizer
from ase.optimize.bfgs import BFGS
from .interpolate_gen import interpolate_custom,interpolate_custom_idpp
import torch
from typing import Optional
import lmdb
from data_processing.reaction_data_creation import reaction_data_object
import pickle
import numpy as np
parser = argparse.ArgumentParser()

# workdir = "proparg"  #
parser.add_argument("--workdir", type=str,default=f"Cat75")
parser.add_argument("--interpolate_type", type=str, default="custom")
parser.add_argument("--transition_state", type=str, default=None)
parser.add_argument("--n_images", type=int, default=8)
parser.add_argument("--output", type=str, default=f"./paired_50/")
parser.add_argument("--neb_fmax", type=float, default=0.25)
parser.add_argument("--cineb_fmax", type=float, default=0.05)
parser.add_argument("--opt_steps", type=int, default=200)
parser.add_argument("--neb_steps", type=int, default=200)
parser.add_argument("--cineb_steps", type=int, default=500)
# parser.add_argument("--opt_model_path", type=str, default="/ssd/liwentao/ocp_main/checkpoints/2024-12-09-21-28-32-generate_cata_opt/best_checkpoint.pt")
parser.add_argument("--further_opt", action="store_true", default=False)
parser.add_argument("--data_path", type=str, default="gen_cata_data/reaction/all")
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--neb_model_path", type=str, default="./final_checkpoints/gen_cata_ckpt/best_checkpoint.pt")
parser.add_argument("--opt_model_path", type=str, default=None)
args = parser.parse_args()

def interpolate_band(args,atom_configs, transition_state=None):
    if transition_state:
        transition_state = read(transition_state)
        ts_positions = transition_state.get_positions()
        middle_idx = len(atom_configs) // 2
        atom_configs[middle_idx].set_positions(ts_positions)
        first_band = NEB(atom_configs[: middle_idx + 1])
        second_band = NEB(atom_configs[middle_idx:])
        first_band.interpolate("idpp")
        second_band.interpolate("idpp")
    
    elif args.interpolate_type == "custom":
        atom_configs = interpolate_custom(atom_configs[0], atom_configs[-1], num_intermediate=args.n_images)
        from ase.io import write
        write(f'./data/{args.data_path}/{args.workdir}/mlneb/{args.workdir}_neb_init_custom.traj', atom_configs)
    
    return atom_configs


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


def get_rxn_list(data_path):
    # 获取目录路径
    base_dir = f"./data/{data_path}"
    
    # 获取所有以Cat开头的文件夹
    rxn_list = []
    for folder in os.listdir(base_dir):
        if folder.startswith('Cat') and os.path.isdir(os.path.join(base_dir, folder)):
            rxn_list.append(folder)
            
    return rxn_list


def main(args,rxn_list):
    args.parent = f"./data/{args.data_path}"
    with open(os.path.join(f"./data/{args.data_path}", "energy_barrier.csv"), "w") as f:
        f.write("reaction,predict_ts_energy,energy_barrier,reactant_energy,product_energy\n")
    # 2024-11-26-10-29-20-cata_e40_ec03_p_mask_m20_e10  2024-12-08-01-12-32-generate_cata_more
    calc = OCPCalculatorMod(checkpoint_path=args.neb_model_path,cpu=False, device_id=args.device_id)
    import time
    start_time = time.time()
    for rxn in rxn_list:
        args.workdir = rxn
        print(f"Processing {args.workdir} ...")
        
        args.reactant = f"./data/{args.data_path}/{args.workdir}/{args.workdir}_reactant.xyz"
        args.product = f"./data/{args.data_path}/{args.workdir}/{args.workdir}_product.xyz"
        args.output = f"./data/{args.data_path}/{args.workdir}/mlneb"
        product = read(args.product)
        reactant = read(args.reactant)
        os.makedirs(args.output, exist_ok=True)
        images = [reactant.copy() for i in range(args.n_images - 1)] + [product]
        

        

        if args.opt_model_path is not None:
            opt_calc = OCPCalculatorMod(checkpoint_path=args.opt_model_path,cpu=False, device_id=args.device_id)

        print("Relaxing endpoints ... ",flush=True)
        # 松弛反应物和产物
        if args.opt_model_path is not None:
            images[0].set_calculator(opt_calc)
            images[-1].set_calculator(opt_calc)
            reactant_opt = BFGS(images[0],logfile=None)
            converge_reactant = reactant_opt.run(fmax=0.06, steps=args.opt_steps)
            product_opt = BFGS(images[-1],logfile=None)
            converge_product = product_opt.run(fmax=0.06, steps=args.opt_steps)
        
            if converge_reactant and converge_product:
                print("成功预优化")

        rp_pair = [images[0].copy()] + [images[-1].copy()]
        images[0].set_calculator(calc)
        images[-1].set_calculator(calc)
        reactant_opt = BFGS(images[0],logfile=None) # ,logfile=None
        converge_reactant = reactant_opt.run(fmax=0.05, steps=args.opt_steps)
        if not converge_reactant:
            images[0] = rp_pair[0].copy()
            images[0].calc = calc
            reactant_opt = BFGS(images[0],logfile=None)
            converge_reactant = reactant_opt.run(fmax=0.05+0.05, steps=args.opt_steps)

        product_opt = BFGS(images[-1],logfile=None)
        converge_product = product_opt.run(fmax=0.05, steps=args.opt_steps)
        if not converge_product:
            images[-1] = rp_pair[-1].copy()
            images[-1].calc = calc
            product_opt = BFGS(images[-1],logfile=None)
            converge_product = product_opt.run(fmax=0.05+0.05, steps=args.opt_steps)

        if converge_reactant and converge_product:
            print("成功优化结构",flush=True)
        else:
            print("优化结构失败 fmax = 0.05, steps = 200",flush=True)
        
        write(os.path.join(args.output, "reactant.xyz"), images[0])
        write(os.path.join(args.output, "reactant.png"), images[0])
        write(os.path.join(args.output, "product.xyz"), images[-1])
        write(os.path.join(args.output, "product.png"), images[-1])

        # try:
        images = interpolate_band(args,images, args.transition_state)
        # except:
        #     print(f"插值失败 for {args.workdir}",flush=True)
        #     continue
        neb= NEB(images, climb=False, method="aseneb",
            allow_shared_calculator=True)
        for image in images:
            image.set_calculator(calc)
        neb_tools = NEBTools(neb.images)

        optimizer = NEBOptimizer(neb,trajectory=None,method='ODE',hmin=0.0005)

        optimizer.run(fmax=args.neb_fmax, steps=args.neb_steps)

        print("NEB has converged, turn on CI-NEB ...")
        neb.climb = True
        converged = optimizer.run(fmax=args.cineb_fmax, steps=args.cineb_steps)

        if converged:
            open(os.path.join(args.output, "converged"), "w")
            print("Reaction converged ... ")


        fig = plot_mep(neb_tools)
        fig.savefig(os.path.join(args.output, "mep.png"))

        energies = [image.get_potential_energy() for image in images]
        # 绘制能量变化折线图
        predict_ts_energy = max(energies)
        predict_ts = images[energies.index(max(energies))]
        print(f"Predict TS energy: {predict_ts_energy:.4f} eV")
        energy_barrier = predict_ts_energy - images[0].get_potential_energy()
        reactant_energy = images[0].get_potential_energy()  
        product_energy = images[-1].get_potential_energy()
        print(f"Energy barrier: {energy_barrier:.4f} eV")
        print(f"Reactant energy: {reactant_energy:.4f} eV")
        print(f"Product energy: {product_energy:.4f} eV")

        write(os.path.join(args.output, f"{args.workdir}_final_neb.xyz"), images)

        with open(os.path.join(f"./data/{args.data_path}", "energy_barrier.csv"), "a") as f:
            f.write(f"{args.workdir},{predict_ts_energy:.4f},{energy_barrier:.4f},{reactant_energy:.4f},{product_energy:.4f}\n")
        write(os.path.join(args.output, "transition_state.xyz"), predict_ts)
        write(os.path.join(args.output, "transition_state.png"), predict_ts)

        write(os.path.join(args.output, f"{args.workdir}_final_neb.traj"), neb.images)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

def plot_mep(neb_tools):
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

    return fig



if __name__ == "__main__":

    rxn_list = get_rxn_list(args.data_path)
    print(rxn_list)
    main(args,rxn_list)
