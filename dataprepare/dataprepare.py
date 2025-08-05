import numpy as np
import torch
import os
from scipy.spatial import KDTree
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from scipy.stats import mode


# Step 1: 从 txt 文件加载点云数据
def load_point_cloud(file_path):
    """
    从 txt 文件中加载点云数据，假设数据格式为：
    每行 [x, y, z, semantic_label, instance_label]
    """
    data = np.loadtxt(file_path)  # 加载点云
    points = data[:, :3]  # 提取坐标: [x, y, z]
    # 坐标归一化
    xyz_min = points.min(axis=0)
    xyz_max = points.max(axis=0)
    xyz_normalized = (points - xyz_min) / (xyz_max - xyz_min)
    semantic_labels = data[:, 6].astype(int)  # 语义标签
    instance_labels = data[:, 7].astype(int)  # 实例标签
    return xyz_normalized, semantic_labels, instance_labels


### Step 3: 计算点的几何特性（50-kd树） ###
def compute_geometric_features(points, k=50):
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


def normalize_features(features):
    """对输入特征进行归一化处理."""
    return (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-6)


def compute_point_curvature(points, k=50):
    """
    计算点云法向量、曲率和密度。
    """
    tree = KDTree(points)
    curvature = np.zeros(points.shape[0])

    for i, point in enumerate(points):
        indices = tree.query(point, k=k)[1]  # 查询 k 近邻
        neighbors = points[indices]

        # 法向量计算
        centroid = neighbors.mean(axis=0)
        cov_matrix = np.cov((neighbors - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)  # 特征分解
        curvature[i] = eigenvalues[0] / eigenvalues.sum()
        # 对特征进行归一化
    curvature = normalize_features(curvature)

    return curvature


def adjusted_edge_weight(point_a, point_b,
                         curvature_a, curvature_b,
                         x_weight=1.0, y_weight=1, z_weight=1.0):
    """
    根据点之间距离、法向量、曲率和密度计算边权重，加入 y 方向权重增强。
    """
    # 几何距离权重，动态调整 y 方向（条纹方向）的重要性
    delta = np.abs(point_a - point_b)
    x_distance = delta[0]
    y_distance = delta[1]
    z_distance = delta[2]
    distance_weight = x_weight * x_distance + y_weight * y_distance + z_weight * z_distance

    # 计算法向量一致性 (夹角余弦相似性)
    # normal_similarity = 1 - np.dot(normal_a, normal_b) / (np.linalg.norm(normal_a) * np.linalg.norm(normal_b) + 1e-6)

    # 曲率差异
    curvature_diff = np.abs(curvature_a - curvature_b)

    # 密度差异
    # density_diff = np.abs(density_a - density_b)

    # 综合加权
    # weight = (distance_weight + 100*curvature_diff + 100*density_diff) / 3
    weight = (distance_weight + 100 * curvature_diff) / 2
    # print(distance_weight,100*curvature_diff,100*density_diff)
    return weight


def build_adaptive_graph(points, curvature, k=50):
    """
    构建改进的加权图，减少线性条纹趋势。
    """
    tree = KDTree(points)
    edges = []

    # 遍历点云，构建邻居权重 (添加方向感知)
    for i, point in enumerate(points):
        dists, indices = tree.query(point, k=k + 1)
        indices = indices[1:]
        dists = dists[1:]

        for j, idx in enumerate(indices):
            weight = adjusted_edge_weight(point, points[idx],
                                          curvature[i], curvature[idx])
            edges.append((i, idx, weight))

    return edges


def felzenszwalb_segmentation(points, edges, scale=6.0, min_size=40):
    """
    改进版本的 Felzenszwalb 分割。
    新增多方向权重处理，平衡区域划分。
    """
    n_points = points.shape[0]
    parent = np.arange(n_points)
    size = np.ones(n_points)
    internal_diff = np.zeros(n_points)

    def find_root(node):
        if parent[node] != node:
            parent[node] = find_root(parent[node])
        return parent[node]

    edges = sorted(edges, key=lambda x: x[2])

    for u, v, weight in edges:
        root_u = find_root(u)
        root_v = find_root(v)

        if root_u != root_v:
            tau_u = internal_diff[root_u] + scale / size[root_u]
            tau_v = internal_diff[root_v] + scale / size[root_v]

            if weight <= tau_u and weight <= tau_v:
                new_root = root_u if size[root_u] > size[root_v] else root_v
                old_root = root_v if new_root == root_u else root_u

                parent[old_root] = new_root
                size[new_root] += size[old_root]
                internal_diff[new_root] = max(internal_diff[root_u], internal_diff[root_v], weight)

    for u, v, weight in edges:
        root_u = find_root(u)
        root_v = find_root(v)

        if root_u != root_v and (size[root_u] < min_size or size[root_v] < min_size):
            new_root = root_u if size[root_u] > size[root_v] else root_v
            old_root = root_v if new_root == root_u else root_u

            parent[old_root] = new_root
            size[new_root] += size[old_root]

    segment_labels = np.array([find_root(i) for i in range(n_points)])
    unique_labels = np.unique(segment_labels)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    return np.array([label_mapping[label] for label in segment_labels])


### Step 4: 提取superpoint的属性特征 ###
def extract_superpoint_features(point_cloud, curvatures):
    superpoint_labels = point_cloud[:, 5].astype(int)  # 获取 superpoints 的标签
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
        mlinearity = sp_points[:,7].mean()
        mplanarity = sp_points[:,8].mean()
        mscattering = sp_points[:,9].mean()
        mverticality = sp_points[:,10].mean()




        sp_feature = np.hstack([sp_id, sp_semantic_mode, sp_instance_mode, sp_center,
                                sp_normal, bounding_box_lwh, sp_density, sp_avg_distance,
                                sp_avg_curvature, sp_point_count,mlinearity, mplanarity, mscattering, mverticality])
        superpoint_features.append(sp_feature)

    return np.array(superpoint_features)


### Step 5: 基于原始50-kd树生成superpoint图结构 ###
def build_superpoint_graph(point_cloud, superpoint_features):
    superpoints = superpoint_features[:, 0].astype(int)  # 获取 superpoint ID
    superpoint_centers = superpoint_features[:, 3:6]
    point_sp_map = point_cloud[:, 5].astype(int)

    # 构建点云50-kd树
    tree = KDTree(point_cloud[:, :3])
    edges = set()

    for i in range(len(point_cloud)):
        _, neighbors = tree.query([point_cloud[i, :3]], k=50)
        for neighbor in neighbors[0]:
            sp1, sp2 = point_sp_map[i], point_sp_map[neighbor]
            if sp1 != sp2:  # 不同的 superpoint 连接
                edges.add((min(sp1, sp2), max(sp1, sp2)))

                # 构建边 index
    edge_index = np.array(list(edges)).T
    return edge_index


def label_to_color(labels, max_colors=1000):
    """
    将超点标签映射为颜色（RGB）。
    使用循环颜色映射方式，保证颜色既不同又可重复利用。

    参数:
    - labels: 超点标签数组（N, ），类型为整数
    - max_colors: 最大可用颜色数

    输出:
    - colors: 每个点的 RGB 颜色值，形状为 (N, 3)，范围 [0, 255]
    """
    num_labels = len(np.unique(labels))  # 超点数
    unique_labels = np.unique(labels)

    # 创建循环颜色映射 (采用 colormap 或随机分配颜色)
    cmap = plt.cm.get_cmap("tab20", max_colors)  # tab20 包含 20 种不同颜色
    base_colors = (cmap(np.arange(max_colors))[:, :3] * 255).astype(int)  # 映射为 RGB

    # 将每个 label 映射到 base_colors 中
    label_to_color_map = {label: base_colors[i % max_colors] for i, label in enumerate(unique_labels)}
    colors = np.array([label_to_color_map[label] for label in labels])

    return colors


def hash_label_to_color(labels):
    """
    使用哈希方法将超点标签映射为伪随机但固定的颜色。

    参数:
    - labels: 超点标签数组 (N, )，类型为整数

    输出:
    - colors: 每个点的 RGB 颜色值，形状为 (N, 3)，范围 [0, 255]
    """
    np.random.seed(42)  # 固定随机种子，保证可复现性
    unique_labels = np.unique(labels)

    colors = np.zeros((len(labels), 3), dtype=np.uint8)

    # 生成颜色映射表
    for label in unique_labels:
        np.random.seed(label)  # 用 label 值作为种子
        colors[labels == label] = np.random.randint(0, 256, size=3)  # 生成固定 RGB

    return colors


def save_to_ply_with_colors(filename, points, labels):
    """
    保存点云到 .ply 文件，附带按 label 编码的颜色。

    参数:
    - filename: 输出 .ply 文件路径
    - points: 点云的 xyz 坐标数组 (N, 3)
    - labels: 每个点的超点标签 (N,)
    """
    print(f"开始保存点云，共有 {len(points)} 点和 {len(np.unique(labels))} 个超点标签...")

    # 将标签映射为 RGB 颜色
    if len(np.unique(labels)) > 1000:
        print("标签数量很大，使用哈希映射生成颜色...")
        colors = hash_label_to_color(labels)  # 使用哈希颜色映射
    else:
        print("标签数量适中，使用循环颜色映射...")
        colors = label_to_color(labels, max_colors=10000)  # 使用循环颜色映射

    # 保存为 .ply 文件
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(points.shape[0]):
            x, y, z = points[i]
            r, g, b = colors[i]
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")

    print(f"点云保存完成：{filename}")


data_root = 'train'
file_list = os.listdir(data_root)
for file_path in file_list:
    print('开始加载' + file_path)
    # file_path = "0070.txt"  # 替换为你的点云路径

    # 加载点云
    points, semantic_labels, instance_labels = load_point_cloud(data_root + '/' + file_path)
    print(file_path + '加载完毕')
    name = file_path.split(".")[0]


    curvatures = compute_point_curvature(points, k=50)
    print('点的曲率计算完成')

    edges = build_adaptive_graph(points, curvatures)
    superpoint_labels = felzenszwalb_segmentation(points, edges, scale=10.0, min_size=50)

    print(superpoint_labels)
    print(superpoint_labels.shape)

    geometric_features = compute_geometric_features(points)
    point_cloud_with_features = np.hstack([points, semantic_labels[:, np.newaxis], instance_labels[:, np.newaxis], superpoint_labels[:, np.newaxis], curvatures[:, np.newaxis], geometric_features])
    print('点的几何特性计算完成')

    # 保存归一化点特征
    np.savetxt("temp/txt/" + name + "_50_10_50.txt", point_cloud_with_features)
    print('保存归一化点特征')


    unique_superpoint_labels, superpoint_labels_counts = np.unique(superpoint_labels, return_counts=True)
    # 输出超点信息

    print("生成的超点标签:", unique_superpoint_labels, superpoint_labels_counts)

    save_to_ply_with_colors("temp/ply/" + name + "_50_10_50.ply", points, superpoint_labels)
    print("生成ply")

    # Step 4: Superpoint 特征提取
    superpoint_features = extract_superpoint_features(point_cloud_with_features, curvatures)
    np.savetxt("temp/superpoint/" + name + "_superpoint_50_10_50.txt", superpoint_features)
    print('Superpoint 特征提取计算完成')


    # Step 5: 构建 superpoint 图
    edge_index = build_superpoint_graph(point_cloud_with_features, superpoint_features)
    node_features = torch.tensor(superpoint_features[:, 1:], dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    graph = Data(x=node_features, edge_index=edge_index)

    # 保存 superpoint 图
    torch.save(graph, "temp/superpoint/" + name + "_superpoint_50_10_50.pt")
    print('保存 superpoint 图')
