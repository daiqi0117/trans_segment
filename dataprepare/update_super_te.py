import os
import numpy as np
from scipy.stats import mode  # 用于计算众数

# 定义文件夹路径
point_cloud_folder = 'output_test'  # 点云文件存放路径
superpoint_folder = 'super_test_50_10_50/superpoint'  # 超点文件存放路径
output_folder = 'updated_superpoints/test/txt'  # 更新后的超点文件存放路径

# 创建输出目录（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取点云文件和超点文件列表
point_cloud_files = sorted([f for f in os.listdir(point_cloud_folder) if f.endswith('.txt')])
superpoint_files = sorted([f for f in os.listdir(superpoint_folder) if f.endswith('.txt')])

# 文件对应关系（假设文件名前四位匹配）
file_map = {sp[:4]: sp for sp in superpoint_files}

# 遍历点云文件
for pc_file in point_cloud_files:
    key = pc_file[:4]  # 获取文件名前四位作为关键标识
    if key not in file_map:
        print(f"没有找到超点文件对应点云文件：{pc_file}")
        continue

    # 文件路径
    pc_file_path = os.path.join(point_cloud_folder, pc_file)
    sp_file_path = os.path.join(superpoint_folder, file_map[key])
    output_file_path = os.path.join(output_folder, file_map[key])

    # 读取点云数据 (n×11 矩阵)
    point_cloud_data = np.loadtxt(pc_file_path)  # 假设点云格式：[x, y, z, semantic_label, instance_label, ...]

    # 读取超点数据 (n×20 矩阵)
    superpoint_data = np.loadtxt(sp_file_path)  # 假设超点格式：[superpoint_id, superpoint_semantic_label, ...]

    # 提取每个点的超点标签和语义标签
    point_superpoint_ids = point_cloud_data[:, 5].astype(int)  # 假设第 4 列为超点标签
    point_semantic_labels = point_cloud_data[:, 3].astype(int)  # 假设第 2 列为语义标签

    # 计算每个超点的语义标签众数
    unique_superpoints = np.unique(point_superpoint_ids)
    superpoint_semantic_map = {}  # 用于存储每个超点的语义标签众数

    for sp_id in unique_superpoints:
        # 找到属于该超点的点的索引
        sp_mask = point_superpoint_ids == sp_id
        sp_labels = point_semantic_labels[sp_mask]

        # 计算该超点的语义标签的众数
        sp_majority_label = mode(sp_labels).mode  # 众数
        superpoint_semantic_map[sp_id] = sp_majority_label

    # 更新超点文件中的语义标签
    for i, row in enumerate(superpoint_data):
        sp_id = int(row[0])  # 假设第 1 列是超点 ID
        if sp_id in superpoint_semantic_map:
            superpoint_data[i, 1] = superpoint_semantic_map[sp_id]  # 更新第 2 列（超点语义标签）

    # 保存更新后的超点数据
    np.savetxt(output_file_path, superpoint_data)

    print(f"已更新超点文件：{file_map[key]}")

print("所有超点文件的语义标签已更新完成！")