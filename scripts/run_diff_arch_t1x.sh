# nohup python ts_search.py \
#  --checkpoint_path "/ssd/liwentao/ocp_main/checkpoints/2024-10-12-10-35-44-painn_t1x/best_checkpoint.pt" \
#  --save_path "./data/val_test_reaction/PaiNN_t1x" --device_id 0 \
#  --dataset_pth "./data/val_test_reaction/reaction.lmdb" \
#   >./experiment_log/PaiNN_t1x_neb.log &

# nohup python ts_search.py \
#  --checkpoint_path "/ssd/liwentao/ocp_main/checkpoints/2024-10-12-12-05-20-schnet_t1x/best_checkpoint.pt" \
#  --save_path "./data/val_test_reaction/Schnet_t1x" --device_id 0 \
#  --dataset_pth "./data/val_test_reaction/reaction.lmdb" \
#   >./experiment_log/Schnet_t1x_neb.log &

python ts_search.py \
 --checkpoint_path "/ssd/liwentao/ocp_main/checkpoints/2024-10-15-01-33-52-gemnet_dT_t1x/best_checkpoint.pt" \
 --save_path "./data/val_test_reaction/Gemnet_dT_t1x_test" --device_id 1 \
 --dataset_pth "./data/val_test_reaction/reaction.lmdb" \
  # >./experiment_log/Gemnet_dT_t1x_neb.log &

# nohup python ts_search.py \
#  --checkpoint_path "/ssd/liwentao/ocp_main/checkpoints/2024-10-15-01-33-52-dpp_t1x/best_checkpoint.pt" \
#  --save_path "./data/val_test_reaction/Dpp_t1x" --device_id 1 \
#  --dataset_pth "./data/val_test_reaction/reaction.lmdb" \
#   >./experiment_log/Dpp_t1x_neb.log &