import numpy as np
from scipy.stats import mode

# 模拟点云数据 (xyz，语义标签，实例标签，超点标签)
point_cloud = np.array([
    [0.1, 0.2, 0.3, 1, 100, 0],   # 点 (xyz，语义标签1，实例标签100，superpoint 0)
    [0.2, 0.3, 0.4, 1, 100, 0],   # 点（同一个superpoint）
    [0.5, 0.6, 0.7, 2, 101, 0],   # 点（superpoint 1）
    [0.6, 0.7, 0.8, 2, 101, 0],
    [0.9, 1.0, 1.1, 3, 101, 0]    # 超点 2
])
curvatures = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # 对应点的曲率

def extract_superpoint_features(point_cloud, curvatures):
    superpoint_labels = point_cloud[:, -1].astype(int)  # 获取 superpoints 的标签
    superpoint_ids = np.unique(superpoint_labels)  # superpoint ID 列表

    superpoint_features = []

    for sp_id in superpoint_ids:
        sp_points = point_cloud[superpoint_labels == sp_id]  # 该 superpoint 内部点
        sp_curvatures = curvatures[superpoint_labels == sp_id]

        # 计算特征
        print(sp_points)
        sp_semantic_mode = mode(sp_points[:, 3]).mode

        sp_instance_mode = mode(sp_points[:, 4]).mode
        sp_center = sp_points[:, :3].mean(axis=0)

        # 法向量（基于 PCA）
        cov_matrix = np.cov(sp_points[:, :3].T)
        print(cov_matrix)
        _, eigenvectors = np.linalg.eigh(cov_matrix)
        sp_normal = eigenvectors[:, -1]
        if sp_normal[2] < 0:  # 保证法向量与 z 轴一致
            sp_normal = -sp_normal
        print(sp_normal)
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

def test_mode():
    """ 测试 mode 函数是否正确处理 """
    semantic_labels = np.array([1, 1, 2, 2, 3])  # 模拟语义标签
    result = mode(semantic_labels, axis=None)
    print(result)
    print(f"Mode result (modern SciPy): {result.mode}")

def test_extract_superpoint_features():
    """ 测试修复后的函数 """
    features = extract_superpoint_features(point_cloud, curvatures)
    print("Extracted superpoint features:")
    print(features)

# 测试
test_mode()
test_extract_superpoint_features()