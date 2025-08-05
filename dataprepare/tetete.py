import os
import numpy as np
from collections import Counter
from tqdm import tqdm


def update_superpoint_file(point_cloud, superpoint_file):
    """
    更新超点文件信息，依据点云文件（更新后的信息）提取超点的语义和实例信息。

    参数:
        point_cloud: numpy.ndarray - 点云文件，n*11 大小。
          第 4 列是语义标签，第 5 列是实例标签，第 6 列是超点 ID。
        superpoint_file: numpy.ndarray - 超点文件，k*20 大小。
          第 1 列是超点 ID，第 2 列是语义标签，第 3 列是实例标签。

    返回:
        numpy.ndarray - 更新语义和实例标签后的超点文件。
    """
    # 提取点云文件中的超点相关列
    superpoint_ids = point_cloud[:, 5].astype(int)  # 超点 ID
    semantic_labels = point_cloud[:, 3].astype(int)  # 语义标签
    instance_labels = point_cloud[:, 4].astype(int)  # 实例标签

    # 初始化存储更新后的超点文件数据
    updated_superpoint_file = superpoint_file.copy()

    for i, row in enumerate(superpoint_file):
        sp_id = int(row[0])  # 超点 ID 位于第 1 列

        # 筛选当前超点的数据点
        mask = superpoint_ids == sp_id
        sp_semantics = semantic_labels[mask]
        sp_instances = instance_labels[mask]

        # 如果当前超点在点云中没有对应点，跳过更新
        if sp_semantics.size == 0 or sp_instances.size == 0:
            continue

            # 统计语义和实例的多数标签
        most_common_semantic = Counter(sp_semantics).most_common(1)[0][0]
        most_common_instance = Counter(sp_instances).most_common(1)[0][0]

        # 更新语义和实例信息
        updated_superpoint_file[i, 1] = most_common_semantic  # 更新语义标签 (第 2 列)
        updated_superpoint_file[i, 2] = most_common_instance  # 更新实例标签 (第 3 列)

    return updated_superpoint_file


def batch_update_superpoint_files(point_cloud_dir, superpoint_dir, output_dir):
    """
    批量更新所有超点文件信息。

    参数:
        point_cloud_dir: str - 点云文件所在文件夹。
        superpoint_dir: str - 超点文件所在文件夹。
        output_dir: str - 更新后的超点文件输出文件夹。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

        # 获取匹配的点云文件和超点文件
    point_cloud_files = {f[:4]: f for f in os.listdir(point_cloud_dir) if f.endswith('.txt')}
    superpoint_files = {f[:4]: f for f in os.listdir(superpoint_dir) if f.endswith('.txt')}

    # 检查所有匹配的文件
    common_files = set(point_cloud_files.keys()) & set(superpoint_files.keys())
    if not common_files:
        print("没有匹配的点云文件和超点文件！")
        return

    for file_key in tqdm(common_files, desc="Updating superpoint files"):
        # 获取当前点云文件和超点文件路径
        point_cloud_path = os.path.join(point_cloud_dir, point_cloud_files[file_key])
        superpoint_path = os.path.join(superpoint_dir, superpoint_files[file_key])

        # 读取点云和超点文件
        point_cloud = np.loadtxt(point_cloud_path)
        superpoint_file = np.loadtxt(superpoint_path)

        # 更新超点文件
        updated_superpoint_file = update_superpoint_file(point_cloud, superpoint_file)

        # 保存更新后的超点文件
        output_path = os.path.join(output_dir, superpoint_files[file_key])
        np.savetxt(output_path, updated_superpoint_file)


if __name__ == "__main__":
    # 定义输入和输出文件夹路径
    point_cloud_dir = "output_processed_test"  # 点云文件夹（更新后的点云文件）
    superpoint_dir = "updated_superpoints/test/txt"  # 超点文件夹
    output_dir = "updated_superpoints_updated_test"  # 输出文件夹

    # 批量更新超点文件
    batch_update_superpoint_files(point_cloud_dir, superpoint_dir, output_dir)