import os
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm


def process_point_cloud(point_cloud):
    """
    对一个点云数组进行处理，确保：
    1. 每个超点的语义和实例标签唯一。
    2. 每个实例的语义标签唯一。

    参数:
        point_cloud: numpy.ndarray - n*11 的点云数组。
          第 4 列为语义标签，第 5 列为实例标签，第 6 列为超点标签。

    返回:
        numpy.ndarray - 处理后的点云数组。
    """
    # 第四列语义标签，第五列实例标签，第六列超点标签
    semantic_col = 3
    instance_col = 4
    superpoint_col = 5

    # Step 1: 确保每个超点的语义标签和实例标签唯一
    superpoint_groups = defaultdict(list)
    for i, point in enumerate(point_cloud):
        superpoint_groups[point[superpoint_col]].append(i)

    for sp_id, indices in superpoint_groups.items():
        # 提取当前超点的数据
        sp_points = point_cloud[indices]

        # 统计语义标签和实例标签
        semantic_labels = Counter(sp_points[:, semantic_col])
        instance_labels = Counter(sp_points[:, instance_col])

        # 确定主语义标签和主实例标签
        main_semantic = semantic_labels.most_common(1)[0][0]
        main_instance = instance_labels.most_common(1)[0][0]

        # 修改超点内所有点的标签
        for idx in indices:
            point_cloud[idx, semantic_col] = main_semantic
            point_cloud[idx, instance_col] = main_instance

            # Step 2: 确保每个实例的语义标签唯一
    instance_groups = defaultdict(list)
    for i, point in enumerate(point_cloud):
        instance_groups[point[instance_col]].append(i)

    for instance_id, indices in instance_groups.items():
        # 提取当前实例组的数据
        instance_points = point_cloud[indices]

        # 统计语义标签
        semantic_labels = Counter(instance_points[:, semantic_col])

        # 确定主语义标签
        main_semantic = semantic_labels.most_common(1)[0][0]

        # 修改实例内的所有点
        for idx in indices:
            point_cloud[idx, semantic_col] = main_semantic

    return point_cloud


def batch_process_files(input_dir, output_dir):
    """
    批处理文件夹内的所有点云文件，对每个点云文件进行标签调整。

    参数:
        input_dir: str - 输入文件夹路径，包含 .txt 点云文件。
        output_dir: str - 输出文件夹路径，存储处理后的点云文件。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # 获取所有 .txt 文件列表
    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    for file in tqdm(files, desc="Processing files"):
        # 读取点云文件
        file_path = os.path.join(input_dir, file)
        point_cloud = np.loadtxt(file_path)

        # 调用处理函数
        processed_point_cloud = process_point_cloud(point_cloud)

        # 保存处理后的点云到新文件夹
        output_path = os.path.join(output_dir, file)
        np.savetxt(output_path, processed_point_cloud)


if __name__ == "__main__":
    # 定义输入输出文件夹
    input_dir = "output_test"
    output_dir = "output_processed_test"

    # 批量处理文件
    batch_process_files(input_dir, output_dir)