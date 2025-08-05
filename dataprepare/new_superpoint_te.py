import os
import numpy as np
import torch

# 定义路径
base_dir = "super_test_50_10_50"  # 根目录
point_cloud_dir = os.path.join(base_dir, "txt")  # 点云文件夹
superpoint_dir = os.path.join(base_dir, "superpoint")  # 超点文件夹
superpoint_graph_dir = os.path.join(base_dir, "pt")  # 超点图文件夹

# 创建输出目录
updated_superpoint_dir = os.path.join(base_dir, "updated_superpoint")  # 更新后的超点文件
updated_superpoint_graph_dir = os.path.join(base_dir, "updated_pt")  # 更新后的超点图文件
os.makedirs(updated_superpoint_dir, exist_ok=True)  # 确保目录存在
os.makedirs(updated_superpoint_graph_dir, exist_ok=True)  # 确保目录存在

# 获取所有文件名称
point_cloud_files = os.listdir(point_cloud_dir)
superpoint_files = os.listdir(superpoint_dir)
superpoint_graph_files = os.listdir(superpoint_graph_dir)

# 遍历点云文件，通过前四位数字匹配其他文件
for point_cloud_file in point_cloud_files:
    if not point_cloud_file.endswith(".txt"):
        continue  # 跳过非 txt 文件

    # 提取点云文件的前四位数字作为文件前缀
    file_prefix = point_cloud_file[:4]  # 假设前四位数字是文件的唯一标识

    # 找到对应的超点文件和超点图文件
    superpoint_file = next(
        (f for f in superpoint_files if f.startswith(file_prefix) and f.endswith(".txt")), None
    )
    superpoint_graph_file = next(
        (f for f in superpoint_graph_files if f.startswith(file_prefix) and f.endswith(".pt")), None
    )

    if not superpoint_file:
        print(f"Warning: Superpoint file for {point_cloud_file} not found.")
        continue

    if not superpoint_graph_file:
        print(f"Warning: Superpoint graph file for {point_cloud_file} not found.")
        continue

    # 构造完整文件路径
    point_cloud_path = os.path.join(point_cloud_dir, point_cloud_file)
    superpoint_path = os.path.join(superpoint_dir, superpoint_file)
    superpoint_graph_path = os.path.join(superpoint_graph_dir, superpoint_graph_file)

    # 加载点云文件 (n*11 矩阵)
    point_cloud = np.loadtxt(point_cloud_path)

    # 加载超点文件 (n*16 矩阵)
    superpoints = np.loadtxt(superpoint_path)

    # 提取点云的超点标签列
    point_superpoint_labels = point_cloud[:, 5]  # 第6列是超点标签

    # 提取需要迁移的4个特征，分别是 linearity, planarity, scattering, verticality
    point_features = point_cloud[:, 7:11]  # 第8到11列是这些特征

    # 计算每个超点中点的特征平均值
    unique_superpoint_ids = np.unique(point_superpoint_labels)  # 获取超点的唯一标签
    new_superpoint_features = []  # 保存每个超点新的4个特征

    # 遍历每个超点 ID 计算平均值
    for superpoint_id in unique_superpoint_ids:
        # 获取所有属于该超点的点的索引
        indices = np.where(point_superpoint_labels == superpoint_id)[0]
        # 计算该超点的 4 个特征（线性度、平面度、散射度、垂直度）的平均值
        mean_features = point_features[indices].mean(axis=0)
        new_superpoint_features.append(mean_features)

    # 将计算的结果转换为 numpy 数组
    new_superpoint_features = np.array(new_superpoint_features)

    # 为超点数据添加新的4个特征
    # 注意超点 ID 的顺序唯一且与 unique_superpoint_ids 对应，因此直接追加
    superpoints = np.hstack((superpoints, new_superpoint_features))



    # 保存更新后的超点文件（txt格式）
    updated_superpoint_file = os.path.join(updated_superpoint_dir, superpoint_file)
    np.savetxt(updated_superpoint_file, superpoints)
    print(updated_superpoint_file)

    # 去掉超点中的语义标签（semantic_label）和实例标签（instance_label）
    # 它们是第2列和第3列（索引1和2）
    superpoints = np.delete(superpoints, [1, 2], axis=1)

    # 更新超点图文件
    # 加载超点图 pt 文件
    superpoint_graph = torch.load(superpoint_graph_path)

    # 更新节点特征，把超点特征替换为新的特征，其中语义标签和实例标签被移除
    # superpoints[:, 2:] 获取去掉 ID（第1列）后的所有新特征
    superpoint_graph.x = torch.tensor(superpoints[:, 1:], dtype=torch.float32)

    # 保存更新后的超点图文件
    updated_superpoint_graph_file = os.path.join(
        updated_superpoint_graph_dir, superpoint_graph_file
    )
    torch.save(superpoint_graph, updated_superpoint_graph_file)
    print(updated_superpoint_graph_file)
    print(f"Processed {point_cloud_file}: Updated superpoint and graph files saved.")

print("所有文件处理完成！")