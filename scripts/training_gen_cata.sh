if [ ! -d "./experiment_log" ]; then
    mkdir ./experiment_log
fi

nohup python main.py --config-yml configs/s2ef/all/equiformer_v2/all_cata/cata_e40_ec03_p_mask_m20_e10_gen.yml \
--mode train --amp --local_rank=1 --checkpoint=./final_checkpoints/cata_ckpt/best_checkpoint.pt \
 --identifier=generate_cata_mix \
  > ./experiment_log/generate_cata_mix.log &
