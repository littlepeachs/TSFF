python -m searching.ts_search --device_id 0 \
--save_path ./save/t1x_neb --checkpoint_path \
./final_checkpoints/t1x_ckpt/best_checkpoint.pt \
--dataset_pth "./data/t1x_data/reaction/reaction.lmdb"

# Set hmin to 0.0012