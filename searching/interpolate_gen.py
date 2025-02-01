import ase
import numpy as np
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.neb import NEB

def interpolate_custom(atoms_r, atoms_p, num_intermediate=10):
    
    atoms_r = rotate_reactant(atoms_r, atoms_p)
    atoms_m = get_middle_structure(atoms_r, atoms_p)

    images1 = [atoms_r.copy() for i in range(num_intermediate//2)] + [atoms_m]

    neb = NEB(images1)
    neb.interpolate("idpp")

    images2 = [atoms_m.copy() for i in range(num_intermediate//2)] + [atoms_p]

    neb = NEB(images2)
    neb.interpolate("idpp")

    return images1[:-1] + images2

def interpolate_custom_idpp(atoms_r, atoms_p, num_intermediate=10):
    atoms_r = rotate_reactant(atoms_r, atoms_p)
    images1 = [atoms_r.copy() for i in range(num_intermediate)] + [atoms_p]
    neb = NEB(images1)
    neb.interpolate("idpp")
    return images1

def rotate_reactant(atoms_1, atoms_2):
    atoms_1 = atoms_1.copy()
    atoms_2 = atoms_2.copy()

    # 获取Rh原子的索引
    Rh_index = atoms_1.get_chemical_symbols().index("Rh")

    cutoffs = natural_cutoffs(atoms_1)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms_1)
    indices, offsets = neighbor_list.get_neighbors(Rh_index)

    # 获取两个P原子的索引
    P_indices = [i for i in indices if atoms_1.get_chemical_symbols()[i] == "P"]
    
    if len(P_indices) != 2:
        raise ValueError("没有找到两个与Rh相连的P原子")

    # 第一步: 旋转一个Rh-P键到共线
    Rh_P1_vector_1 = atoms_1.positions[P_indices[0]] - atoms_1.positions[Rh_index]
    Rh_P1_vector_2 = atoms_2.positions[P_indices[0]] - atoms_2.positions[Rh_index]

    Rh_P1_vector_1 /= np.linalg.norm(Rh_P1_vector_1)
    Rh_P1_vector_2 /= np.linalg.norm(Rh_P1_vector_2)

    rotation_axis_1 = np.cross(Rh_P1_vector_1, Rh_P1_vector_2)
    rotation_axis_1 /= np.linalg.norm(rotation_axis_1)
    angle_1 = np.arccos(np.dot(Rh_P1_vector_1, Rh_P1_vector_2))
    angle_degrees_1 = np.degrees(angle_1)

    atoms_1.rotate(angle_degrees_1, rotation_axis_1, center=atoms_1.positions[Rh_index])

    # 第二步: 旋转整个分子使P-Rh-P平面对齐
    P1_pos_1 = atoms_1.positions[P_indices[0]]
    P2_pos_1 = atoms_1.positions[P_indices[1]]
    P1_pos_2 = atoms_2.positions[P_indices[0]]
    P2_pos_2 = atoms_2.positions[P_indices[1]]

    normal_1 = np.cross(P1_pos_1 - atoms_1.positions[Rh_index], P2_pos_1 - atoms_1.positions[Rh_index])
    normal_2 = np.cross(P1_pos_2 - atoms_2.positions[Rh_index], P2_pos_2 - atoms_2.positions[Rh_index])

    normal_1 /= np.linalg.norm(normal_1)
    normal_2 /= np.linalg.norm(normal_2)

    rotation_axis_2 = np.cross(normal_1, normal_2)
    rotation_axis_2 /= np.linalg.norm(rotation_axis_2)
    angle_2 = np.arccos(np.dot(normal_1, normal_2))
    angle_degrees_2 = np.degrees(angle_2)

    atoms_1.rotate(angle_degrees_2, rotation_axis_2, center=atoms_1.positions[Rh_index])
    # 添加平移操作
    Rh_displacement = atoms_2.positions[Rh_index] - atoms_1.positions[Rh_index]
    atoms_1.translate(Rh_displacement)

    return atoms_1


def get_middle_structure(atoms_1, atoms_2):

    atoms_1 = atoms_1.copy()
    atoms_2 = atoms_2.copy()

    # get index of center atom Rh
    Rh_index = atoms_1.get_chemical_symbols().index("Rh")

    cutoffs = natural_cutoffs(atoms_1)

    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms_1)


    # Get the neighbors of the Rh atom
    indices, offsets = neighbor_list.get_neighbors(Rh_index)

    # determine indices of H atoms
    for i in indices:
        if atoms_1.get_chemical_symbols()[i] == "H":
            H_index = i

    # determine which two indices are from C=C
    C_index = []
    for i in indices:
        if atoms_1.get_chemical_symbols()[i] == "C":
            # check if this C is connected to another C
            neighbor_indices, _ = neighbor_list.get_neighbors(i)
            for j in neighbor_indices:
                if atoms_1.get_chemical_symbols()[j] == "C":
                    C_index.append(i)
                    break

    # get all indices of C2H4 molecule defined by C=C
    C2H4_index = []
    for i in C_index:
        neighbor_indices, _ = neighbor_list.get_neighbors(i)
        for j in neighbor_indices:
            if atoms_1.get_chemical_symbols()[j] == "H":
                C2H4_index.append(j)
            if atoms_1.get_chemical_symbols()[j] == "C":
                C2H4_index.append(j)

    C2H4_index = list(set(C2H4_index))

    # determine which C is closer to H atoms in product
    C0H = atoms_2.get_distance(C_index[0], H_index)
    C1H = atoms_2.get_distance(C_index[1], H_index)

    if C0H > C1H:
        C_up_index = C_index[0]
        C_down_index = C_index[1]
    else:
        C_up_index = C_index[1]
        C_down_index = C_index[0]

    # rotate the C2H4 molecule to make C_up_index closer to H atom
    # based on the line connecting Rh and middle of C=C bond
    # code here

    # Find the midpoint of the C=C bond
    C_pos1 = atoms_1.positions[C_index[0]]
    C_pos2 = atoms_1.positions[C_index[1]]
    C_midpoint = (C_pos1 + C_pos2) / 2

    # Vector from Rh to midpoint of C=C bond
    Rh_pos = atoms_1.positions[Rh_index]
    axis_vector = C_midpoint - Rh_pos
    axis_vector /= np.linalg.norm(axis_vector)  # Normalize the axis vector

    # Determine the direction from the midpoint to the C_up_index atom
    C_up_pos = atoms_1.positions[C_up_index]
    mid_to_C_up = C_up_pos - C_midpoint
    mid_to_C_up /= np.linalg.norm(mid_to_C_up)  # Normalize

    # Determine the direction from the midpoint to the H atom in the product
    H_pos_p = atoms_2.positions[H_index]
    C_up_pos_p = atoms_2.positions[C_up_index]
    mid_to_H_p = H_pos_p - C_up_pos_p
    mid_to_H_p /= np.linalg.norm(mid_to_H_p)  # Normalize

    # Calculate the angle between mid_to_C_up and mid_to_H_p
    angle = -30

    # Determine the rotation direction (sign of the angle)
    cross_prod = np.cross(mid_to_C_up, mid_to_H_p)
    if np.dot(cross_prod, axis_vector) < 0:
        angle = -angle
    # Rotate the selected atoms
    atoms_r1 = atoms_1.copy()
    atoms_r1.rotate(
        a=angle,
        v=axis_vector,
        center=C_midpoint,
        rotate_cell=False,
    )
    # 更新旋转后的C2H4原子位置
    for i, index in enumerate(C2H4_index):
        atoms_1.positions[index] = atoms_r1.positions[C2H4_index[i]]

    # 计算Rh-H键的向量
    Rh_H_vector = atoms_1.positions[H_index] - atoms_1.positions[Rh_index]
    Rh_H_length = np.linalg.norm(Rh_H_vector)
    Rh_H_direction = Rh_H_vector / Rh_H_length

    # 计算平移向量（半个Rh-H键的长度）
    translation_vector = 0.3 * Rh_H_length * Rh_H_direction

    # 只平移C2H4分子
    for index in C2H4_index:
        atoms_1.positions[index] += translation_vector
    # 保持Rh_H_vector的长度不变，将Rh_H_vector向C2H4中的C原子旋转移动，使得H原子接近C_up_index
    # 计算C_up原子的位置
    C_up_pos = atoms_1.positions[C_up_index]
    C_down_pos = atoms_1.positions[C_down_index]
    
    # 计算C_down到C_up的向量
    C_down_to_C_up = C_up_pos - C_down_pos
    C_down_to_C_up_normalized = C_down_to_C_up / np.linalg.norm(C_down_to_C_up)
    
    # 在C_down到C_up的延长线上选择一个点
    extension_factor = 0.8  # 可以根据需要调整这个因子
    extended_point = C_up_pos + extension_factor * C_down_to_C_up_normalized
    
    # 计算从Rh到延长线上点的向量
    Rh_to_extended_point = extended_point - atoms_1.positions[Rh_index]
    Rh_to_extended_point_normalized = Rh_to_extended_point / np.linalg.norm(Rh_to_extended_point)
    
    # 计算新的旋转轴（Rh_H_vector和Rh_to_extended_point的叉积）
    rotation_axis = np.cross(Rh_H_vector, Rh_to_extended_point)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # 计算旋转角度（Rh_H_vector和Rh_to_extended_point之间的夹角）
    angle = np.arccos(np.dot(Rh_H_direction, Rh_to_extended_point_normalized)) * 0.1
    
    # 使用Rodrigues旋转公式旋转Rh_H_vector
    rotated_Rh_H = Rh_H_vector * np.cos(angle) + \
                   np.cross(rotation_axis, Rh_H_vector) * np.sin(angle) + \
                   rotation_axis * np.dot(rotation_axis, Rh_H_vector) * (1 - np.cos(angle))
    
    # 计算H原子的新位置
    new_H_pos = atoms_1.positions[Rh_index] + rotated_Rh_H
    
    # 更新H原子的位置
    atoms_1.positions[H_index] = new_H_pos
    # 将Rh-H键向量延长1.5倍
    extended_Rh_H_vector = rotated_Rh_H * 1.2
    
    # 计算H原子的新位置
    new_H_pos = atoms_1.positions[Rh_index] + extended_Rh_H_vector
    
    # 更新H原子的位置
    atoms_1.positions[H_index] = new_H_pos

    # 计算其他原子的平均位置
    other_indices = [i for i in range(len(atoms_1)) if i not in [Rh_index, H_index] + C2H4_index]
    for i in other_indices:
        atoms_1.positions[i] = (atoms_1.positions[i] + atoms_2.positions[i]) / 2

    # 统计C_up_atom的两个H，并把它们的index存在列表里面
    C_up_H_indices = []
    for i, atom in enumerate(atoms_1):
        if atom.symbol == 'H':
            distance = np.linalg.norm(atom.position - atoms_1.positions[C_up_index])
            if distance < 1.5:  # 假设C-H键长小于1.5埃
                C_up_H_indices.append(i)
    
    # 确保我们找到了两个H原子
    if len(C_up_H_indices) != 2:
        print(f"警告：找到了{len(C_up_H_indices)}个与C_up相连的H原子，而不是预期的2个。")

    # 计算R-C_down向量
    R_C_down_vector = atoms_1.positions[C_down_index] - atoms_1.positions[Rh_index]
    
    # 计算C_down-C_up向量
    C_down_C_up_vector = atoms_1.positions[C_up_index] - atoms_1.positions[C_down_index]
    
    # 计算叉积得到旋转轴
    rotation_axis = np.cross(R_C_down_vector, C_down_C_up_vector)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
    # 对C_up及其两个H原子进行旋转
    # 计算C_up相对于C_down的位置向量
    C_up_relative_pos = atoms_1.positions[C_up_index] - atoms_1.positions[C_down_index]
    
    # 定义旋转角度（可以根据需要调整）
    angle = np.radians(-10)  # 10度
    
    # 使用Rodrigues旋转公式旋转C_up
    rotated_C_up_pos = (C_up_relative_pos * np.cos(angle) + 
                        np.cross(rotation_axis, C_up_relative_pos) * np.sin(angle) + 
                        rotation_axis * np.dot(rotation_axis, C_up_relative_pos) * (1 - np.cos(angle)))
    
    # 更新C_up原子的位置
    atoms_1.positions[C_up_index] = atoms_1.positions[C_down_index] + rotated_C_up_pos
    
    # 对C_up的两个H原子进行旋转
    for H_index in C_up_H_indices:
        # 计算H原子相对于C_down的位置向量
        H_relative_pos = atoms_1.positions[H_index] - atoms_1.positions[C_down_index]
        
        # 使用Rodrigues旋转公式旋转H原子
        rotated_H_pos = (H_relative_pos * np.cos(angle) + 
                         np.cross(rotation_axis, H_relative_pos) * np.sin(angle) + 
                         rotation_axis * np.dot(rotation_axis, H_relative_pos) * (1 - np.cos(angle)))
        
        # 更新H原子的位置
        atoms_1.positions[H_index] = atoms_1.positions[C_down_index] + rotated_H_pos
    
    # print("C_up及其两个H原子已经围绕新的旋转轴旋转。")

    # 构建C_up-C_down旋转轴
    C_up_C_down_axis = atoms_1.positions[C_up_index] - atoms_1.positions[C_down_index]
    C_up_C_down_axis = C_up_C_down_axis / np.linalg.norm(C_up_C_down_axis)

    # 定义旋转角度（顺时针旋转，使用负角度）
    rotation_angle = np.radians(-10)  # 可以根据需要调整角度

    # 对C_up的两个H原子进行旋转
    for H_index in C_up_H_indices:
        # 计算H原子相对于C_up的位置向量
        H_relative_pos = atoms_1.positions[H_index] - atoms_1.positions[C_up_index]
        
        # 使用Rodrigues旋转公式旋转H原子
        rotated_H_pos = (H_relative_pos * np.cos(rotation_angle) + 
                         np.cross(C_up_C_down_axis, H_relative_pos) * np.sin(rotation_angle) + 
                         C_up_C_down_axis * np.dot(C_up_C_down_axis, H_relative_pos) * (1 - np.cos(rotation_angle)))
        
        # 更新H原子的位置
        atoms_1.positions[H_index] = atoms_1.positions[C_up_index] + rotated_H_pos

    # print("C_up的两个H原子已经围绕C_up-C_down轴顺时针旋转。")

    return atoms_1

if __name__ == "__main__":
    import os
    
    # 读取所有以Cat开头的文件夹
    base_dir = "./final_data/opt_result_only_final"
    name = []
    for folder in os.listdir(base_dir):
        if folder.startswith("Cat") and os.path.isdir(os.path.join(base_dir, folder)):
            name.append(folder)
    name.sort()
    for n in name:
        try:
            xyz_file = f"{base_dir}/{n}/{n}_reactant.xyz"
            atoms_r = read(xyz_file, ":")[0]

            xyz_file = f"{base_dir}/{n}/{n}_product.xyz" 
            atoms_p = read(xyz_file, ":")[0]

            images = interpolate_custom(atoms_r, atoms_p, num_intermediate=8)
            from ase.io import write

            write(f"{base_dir}/{n}/{n}_neb_init_custom.traj", images)
            print(f"成功处理 {n}")
        except Exception as e:
            print(f"处理 {n} 时出错: {str(e)}")






