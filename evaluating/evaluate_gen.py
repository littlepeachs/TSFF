import pandas as pd

# 读取sort_barrier_all.csv文件
df = pd.read_csv('./data/gen_cata_data/reaction/sort_barrier_all.csv')

# 筛选energy_barrier大于0.25的反应并取前50个
top50 = df[df['energy_barrier'] > 0.25][:50]

# 创建一个新的DataFrame来存储结果
result_df = pd.DataFrame(columns=['Reaction', 'Barrier', 'Reactant Energy', 'TS Energy'])

# 遍历每个反应
for _, row in top50.iterrows():
    # 添加到结果DataFrame
    result_df = result_df.append({
        'Reaction': row['reaction'],
        'Barrier': row['energy_barrier'],
        'Reactant Energy': row['reactant_energy'], 
        'TS Energy': row['predict_ts_energy']
    }, ignore_index=True)

# 按能垒排序
result_df = result_df.sort_values('Barrier')

# 保存到CSV文件
result_df.to_csv('./data/gen_cata_data/reaction/pred_barrier_50.csv', index=False)

print("已将top50反应数据保存到gt_barrier_50.csv")

# 读取预测和真实值的CSV文件
pred_df = pd.read_csv('./data/gen_cata_data/reaction/pred_barrier_50.csv')
gt_df = pd.read_csv('./data/gen_cata_data/reaction/gt_barrier_50.csv')

# 获取反应名称的交集
common_reactions = set(pred_df['Reaction']).intersection(set(gt_df['Reaction']))
# 创建一个包含common_reactions的DataFrame
common_df = pd.DataFrame(columns=['Reaction', 'GT_Barrier'])

# 获取真实能垒值并添加到DataFrame
for reaction in common_reactions:
    gt_barrier = gt_df[gt_df['Reaction'] == reaction]['Barrier'].values[0]
    common_df = common_df.append({
        'Reaction': reaction,
        'GT_Barrier': gt_barrier
    }, ignore_index=True)

# 按真实能垒排序
common_df = common_df.sort_values('GT_Barrier')

# 更新common_reactions为排序后的列表
common_reactions = common_df['Reaction'].tolist()

print(f"预测值和真实值的交集大小: {len(common_reactions)}")

# 计算交集中反应的MAE
mae_list = []
mae_dict = {}
gt_barrier_dict = {}
for reaction in common_reactions:
    pred_barrier = pred_df[pred_df['Reaction'] == reaction]['Barrier'].values[0]
    gt_barrier = gt_df[gt_df['Reaction'] == reaction]['Barrier'].values[0]
    mae = abs(pred_barrier - gt_barrier)
    mae_list.append(mae)
    mae_dict[reaction] = mae
    gt_barrier_dict[reaction] = gt_barrier
    # print(f"{reaction} 的MAE: {mae:.4f}")

avg_mae = sum(mae_list) / len(mae_list)
std_mae = (sum((x - avg_mae) ** 2 for x in mae_list) / len(mae_list)) ** 0.5
mae_25 = sorted(mae_list)[int(len(mae_list) * 0.25)]
mae_50 = sorted(mae_list)[int(len(mae_list) * 0.50)]
mae_75 = sorted(mae_list)[int(len(mae_list) * 0.75)]

print(f"\n平均MAE: {avg_mae:.4f}")
print(f"MAE标准差: {std_mae:.4f}")
print(f"MAE 25分位数: {mae_25:.4f}")
print(f"MAE 50分位数: {mae_50:.4f}")
print(f"MAE 75分位数: {mae_75:.4f}")


import os
import numpy as np
from ase.io import read
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def kabsch_rmsd(P, Q):
    """
    使用Kabsch算法计算RMSD
    P, Q 是两个N*3的坐标矩阵
    """
    # 计算质心
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # 将坐标移到质心
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # 计算协方差矩阵
    H = P_centered.T @ Q_centered
    
    # SVD分解
    U, S, Vt = np.linalg.svd(H)
    
    # 计算旋转矩阵
    R = Vt.T @ U.T
    
    # 如果行列式为负,需要修正以避免镜像
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    # 将P旋转到Q的坐标系
    P_rotated = P_centered @ R + centroid_Q
    
    # 计算RMSD
    rmsd = np.sqrt(np.mean(np.sum((P_rotated - Q)**2, axis=1)))
    return rmsd

print("\n计算不同方法预测的TS结构的RMSD...")

rmsd_list = []
rmsd_dict = {}
for reaction in common_reactions:
    # 检查P_H和P_wo_H文件夹
    found = False
    gen_path = os.path.join('./data/gen_cata_data/reaction/', "all", reaction, 'mlneb', 'transition_state.xyz')
    if os.path.exists(gen_path):
        found = True
        gen_ts = read(gen_path)
        final_ts = read(os.path.join('./data/gen_cata_data/final_gen', reaction, 'neb', 'transition_state.xyz'))
        
        # 获取坐标
        gen_pos = gen_ts.get_positions()
        final_pos = final_ts.get_positions()
        
        # 计算RMSD
        rmsd = kabsch_rmsd(gen_pos, final_pos)
        rmsd_list.append(rmsd)
        rmsd_dict[reaction] = rmsd
        # print(f"{reaction} 的RMSD: {rmsd:.4f}")
            
    if not found:
        print(f"警告: 未找到{reaction}的TS结构文件")

if rmsd_list:
    avg_rmsd = np.mean(rmsd_list)
    std_rmsd = np.std(rmsd_list)
    rmsd_25 = np.percentile(rmsd_list, 25)
    rmsd_50 = np.percentile(rmsd_list, 50)
    rmsd_75 = np.percentile(rmsd_list, 75)
    
    print(f"\n平均RMSD: {avg_rmsd:.4f}")
    print(f"RMSD标准差: {std_rmsd:.4f}")
    print(f"RMSD 25分位数: {rmsd_25:.4f}")
    print(f"RMSD 50分位数: {rmsd_50:.4f}")
    print(f"RMSD 75分位数: {rmsd_75:.4f}")
