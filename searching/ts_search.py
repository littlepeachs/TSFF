import argparse
import sys
sys.path.append('../data_processing')
from .neb_predictor import MLFF_NEB_Predictor
import os
from data_processing.reaction_data_creation import reaction_data_object
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--neb_config_path', type=str,default="./configs/neb_config/neb.yml")
    parser.add_argument('--cineb_config_path', type=str,default="./configs/neb_config/cineb.yml")
    parser.add_argument('--checkpoint_path', type=str, default="./final_checkpoints/t1x_ckpt/best_checkpoint.pt") # 2024-06-03-20-30-56
    parser.add_argument('--save_path', type=str, default="./save/t1x_neb")
    parser.add_argument('--save_lmdb_size', type=int, default=1099511627776 * 2)
    parser.add_argument('--dataset_pth', type=str, default="./data/t1x_data/reaction/reaction.lmdb")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--total_chunk', type=int, default=1)
    parser.add_argument('--chunk', type=int, default=0)
    parser.add_argument('--cpu', type=bool, default=False)
    parser.add_argument("--type",type=str,default="neb")
    parser.add_argument("--opt_model_path",type=str,default=None) # /ssd/liwentao/ocp_main/checkpoints/2024-11-29-00-21-20-cata_e40_ec03_p_mask_m20_e10_opt/best_checkpoint.pt
    args = parser.parse_args()
    
    
    if not os.path.exists(args.save_path+"/neb"):
        os.makedirs(args.save_path+"/neb")
    if not os.path.exists(args.save_path+"/neb_ci"):
        os.makedirs(args.save_path+"/neb_ci")
    if not os.path.exists(args.save_path+"/neb_init"):
        os.makedirs(args.save_path+"/neb_init")
    if not os.path.exists(args.save_path+"/mep_fig"):
        os.makedirs(args.save_path+"/mep_fig")
    
    mnp = MLFF_NEB_Predictor(args,
                                neb_config_path=args.neb_config_path,
                            cineb_config_path=args.cineb_config_path,   
                            checkpoint_path=args.checkpoint_path,
                            save_path=args.save_path,
                            save_lmdb_size=args.save_lmdb_size,
                            dataset_pth=args.dataset_pth,
                            device_id=args.device_id,
                            cpu=args.cpu,
                            total_chunk=args.total_chunk,
                            chunk=args.chunk, )
    
    start_time = time.time()
    mnp.main()
    end_time = time.time()
    print("Time elapsed: ", end_time-start_time)
