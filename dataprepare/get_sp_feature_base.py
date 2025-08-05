import numpy as np
from sklearn.neighbors import KDTree
from scipy.stats import mode
import torch
from torch_geometric.data import Data


### Step 1: 加载点云数据并归一化 ###
def normalize_point_cloud_and_labels(file_path):
    # 加载点云数据 (n * 6): xyz, semantic_label, instance_label, superpoint_label
    point_cloud = np.loadtxt(file_path)
    xyz = point_cloud[:, :3]  # 坐标列
    labels = point_cloud[:, 3:]  # 标签列 (语义、实例、超点标签)

    # 坐标归一化
    xyz_min = xyz.min(axis=0)
    xyz_max = xyz.max(axis=0)
    xyz_normalized = (xyz - xyz_min) / (xyz_max - xyz_min)

    # 返回归一化后的点云数据
    normalized_point_cloud = np.hstack([xyz_normalized, labels])
    return normalized_point_cloud


### Step 2: 基于16-kd树计算每个点的曲率 ###
def compute_curvature(points, k=16):
    # 构建 kd 树并查找 k 近邻
    tree = KDTree(points)

    curvatures = []
    for i in range(len(points)):
        _, idx = tree.query([points[i]], k=k)
        neighbors = points[idx[0]]

        # 计算协方差矩阵，以此估算曲率
        cov_matrix = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        curvature = eigenvalues[0] / np.sum(eigenvalues)  # 曲率为最小特征值 / 特征值和
        curvatures.append(curvature)

    return np.array(curvatures)


### Step 3: 计算点的几何特性（50-kd树） ###
def compute_geometric_features(points, k=16):
    tree = KDTree(points)
    features = []

    for i in range(len(points)):
        _, idx = tree.query([points[i]], k=k)
        neighbors = points[idx[0]]

        # PCA 分解
        cov_matrix = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
        l1, l2, l3 = np.sort(eigenvalues)[::-1]

        # 几何特性
        linearity = (l1 - l2) / l1
        planarity = (l2 - l3) / l1
        scattering = l3 / l1
        verticality = np.abs(np.dot([0, 0, 1], _[:, -1]))  # 与 z 轴对齐的第一主成分

        features.append([linearity, planarity, scattering, verticality])

    return np.array(features)


### Step 4: 提取superpoint的属性特征 ###
def extract_superpoint_features(point_cloud, curvatures):
    superpoint_labels = point_cloud[:, -1].astype(int)  # 获取 superpoints 的标签
    superpoint_ids = np.unique(superpoint_labels)  # superpoint ID 列表

    superpoint_features = []

    for sp_id in superpoint_ids:
        sp_points = point_cloud[superpoint_labels == sp_id]  # 该 superpoint 内部点
        sp_curvatures = curvatures[superpoint_labels == sp_id]

        # 计算特征
        sp_semantic_mode = mode(sp_points[:, 3]).mode
        sp_instance_mode = mode(sp_points[:, 4]).mode
        sp_center = sp_points[:, :3].mean(axis=0)

        # 法向量（基于 PCA）
        cov_matrix = np.cov(sp_points[:, :3].T)
        _, eigenvectors = np.linalg.eigh(cov_matrix)
        sp_normal = eigenvectors[:, -1]
        if sp_normal[2] < 0:  # 保证法向量与 z 轴一致
            sp_normal = -sp_normal

            # 边界框特性
        bounding_box_min = sp_points[:, :3].min(axis=0)
        bounding_box_max = sp_points[:, :3].max(axis=0)
        bounding_box_lwh = bounding_box_max - bounding_box_min

        # 点密度
        volume = np.prod(bounding_box_lwh)
        sp_density = len(sp_points) / volume if volume > 0 else 0

        # 平均欧氏距离
        distances_to_center = np.linalg.norm(sp_points[:, :3] - sp_center, axis=1)
        sp_avg_distance = distances_to_center.mean()

        # 内部点曲率均值与数量
        sp_avg_curvature = sp_curvatures.mean()
        sp_point_count = len(sp_points)

        sp_feature = np.hstack([sp_id, sp_semantic_mode, sp_instance_mode, sp_center,
                                sp_normal, bounding_box_lwh, sp_density, sp_avg_distance,
                                sp_avg_curvature, sp_point_count])
        superpoint_features.append(sp_feature)

    return np.array(superpoint_features)


### Step 5: 基于原始16-kd树生成superpoint图结构 ###
def build_superpoint_graph(point_cloud, superpoint_features):
    superpoints = superpoint_features[:, 0].astype(int)  # 获取 superpoint ID
    superpoint_centers = superpoint_features[:, 3:6]
    point_sp_map = point_cloud[:, -1].astype(int)

    # 构建点云16-kd树
    tree = KDTree(point_cloud[:, :3])
    edges = set()

    for i in range(len(point_cloud)):
        _, neighbors = tree.query([point_cloud[i, :3]], k=16)
        for neighbor in neighbors[0]:
            sp1, sp2 = point_sp_map[i], point_sp_map[neighbor]
            if sp1 != sp2:  # 不同的 superpoint 连接
                edges.add((min(sp1, sp2), max(sp1, sp2)))

                # 构建边 index
    edge_index = np.array(list(edges)).T
    return edge_index


### 主函数：整合所有步骤 ###
def main():
    input_file = "0070_10_100.txt"

    # Step 1: 坐标归一化
    point_cloud = normalize_point_cloud_and_labels(input_file)
    print('坐标归一化计算完成')
    # Step 2: 计算点的曲率
    curvatures = compute_curvature(point_cloud[:, :3])
    print('点的曲率计算完成')
    # Step 3: 几何特性计算
    geometric_features = compute_geometric_features(point_cloud[:, :3])
    point_cloud_with_features = np.hstack([point_cloud, curvatures[:, np.newaxis], geometric_features])
    print('点的几何特性计算完成')
    # 保存归一化点特征
    np.savetxt('point_cloud_with_features.txt', point_cloud_with_features)
    print('保存归一化点特征')
    # Step 4: Superpoint 特征提取
    superpoint_features = extract_superpoint_features(point_cloud, curvatures)
    np.savetxt('superpoint_features.txt', superpoint_features)
    print('Superpoint 特征提取计算完成')
    # Step 5: 构建 superpoint 图
    edge_index = build_superpoint_graph(point_cloud, superpoint_features)
    node_features = torch.tensor(superpoint_features[:, 1:], dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    graph = Data(x=node_features, edge_index=edge_index)

    # 保存 superpoint 图
    torch.save(graph, 'superpoint_graph.pt')
    print('保存 superpoint 图')

if __name__ == "__main__":
    main()