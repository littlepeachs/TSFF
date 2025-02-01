# 原始模型的有H无H大规模搜索，原始模型做优化和搜索
nohup python neb_direct.py \
 --neb_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-14-09-40-16-generate_cata_mix/best_checkpoint.pt" \
 --data_path "generate_valid_structure_mix/P_H" \
 --device_id 1 \
 --neb_steps 200 \
 --cineb_steps 500 \
 --opt_steps 200 \
  > ./experiment_log/P_H_mix_model.log &

nohup python neb_direct.py \
 --neb_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-14-09-40-16-generate_cata_mix/best_checkpoint.pt" \
 --data_path "generate_valid_structure_mix/P_wo_H" \
 --device_id 2 \
 --neb_steps 200 \
 --cineb_steps 500 \
 --opt_steps 200 \
  > ./experiment_log/P_wo_H_mix_model.log &

# # 统一搜索模型的有H无H大规模搜索，原始模型做优化和搜索
# nohup python neb_direct.py \
#  --neb_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-08-01-12-32-generate_cata_more/best_checkpoint.pt" \
#  --data_path "generate_valid_structure_uni/P_H" \
#  --device_id 3 \
#  --neb_steps 200 \
#  --cineb_steps 500 \
#  --opt_steps 200 \
#   > ./experiment_log/P_H_uni_model.log &

# nohup python neb_direct.py \
#  --neb_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-08-01-12-32-generate_cata_more/best_checkpoint.pt" \
#  --data_path "generate_valid_structure_uni/P_wo_H" \
#  --device_id 5 \
#  --neb_steps 200 \
#  --cineb_steps 500 \
#  --opt_steps 200 \
#   > ./experiment_log/P_wo_H_uni_model.log &

# # 优化+搜索模型的有H无H大规模搜索，原始模型做优化和搜索
# nohup python neb_direct.py \
#  --neb_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-08-01-12-32-generate_cata_more/best_checkpoint.pt" \
#  --opt_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-09-21-28-32-generate_cata_opt/best_checkpoint.pt" \
#  --data_path "generate_valid_structure_opt_neb/P_H" \
#  --device_id 6 \
#  --neb_steps 200 \
#  --cineb_steps 500 \
#  --opt_steps 200 \
#   > ./experiment_log/P_H_opt_neb_model.log &

# nohup python neb_direct.py \
#  --neb_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-08-01-12-32-generate_cata_more/best_checkpoint.pt" \
#  --opt_model_path "/ssd/liwentao/ocp_main/checkpoints/2024-12-09-21-28-32-generate_cata_opt/best_checkpoint.pt" \
#  --data_path "generate_valid_structure_opt_neb/P_wo_H" \
#  --device_id 7 \
#  --neb_steps 200 \
#  --cineb_steps 500 \
#  --opt_steps 200 \
#   > ./experiment_log/P_wo_H_opt_neb_model.log &


