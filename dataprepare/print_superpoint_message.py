import numpy as np
from collections import defaultdict


def parse_superpoint_file(file_path):
    """
    解析超点文件，统计每个实例标签下对应的语义标签以及每种语义标签的个数
    Args:
        file_path (str): 超点文件的路径
    Returns:
        dict: 实例分割标签对应的语义标签统计结果
    """
    # 加载txt文件为矩阵（假设文件内容为n*16的格式，以空格或其他符号分隔）
    data = np.loadtxt(file_path)

    # 提取所需的列
    instance_labels = data[:, 2].astype(int)  # 第3列：实例分割标签
    semantic_labels = data[:, 1].astype(int)  # 第2列：语义分割标签
    num = data[:, -1].astype(int)  # 第2列：语义分割标签
    # 字典存储每个实例分割标签下语义标签的统计
    instance_to_semantic = defaultdict(lambda: defaultdict(int))
    instance_to_semantic_num = defaultdict(lambda: defaultdict(int))
    # 遍历每个超点并进行统计
    for instance_label, semantic_label,num_points in zip(instance_labels, semantic_labels,num):
        instance_to_semantic[instance_label][semantic_label] += 1
        instance_to_semantic_num[instance_label][semantic_label] += num_points
    return instance_to_semantic,instance_to_semantic_num


def print_results(instance_to_semantic,instance_to_semantic_num):
    """
    打印每个实例标签下的语义标签及其数量
    Args:
        instance_to_semantic (dict): 实例分割标签对应的语义标签统计结果
    """
    for instance_label, semantic_counts in instance_to_semantic.items():
        print(f"实例标签 {instance_label}:")
        for semantic_label, count in semantic_counts.items():
            print(f"    语义标签 {semantic_label}: {count} 个")
        # 文件路径
    for instance_label, semantic_counts in instance_to_semantic_num.items():
        print(f"实例标签 {instance_label}:")
        for semantic_label, count in semantic_counts.items():
            print(f"    语义标签 {semantic_label}: {count} 个")

file_path = "updated_superpoints/train/txt2/0070_superpoint_50_10_50.txt"

# 解析文件并统计结果
instance_to_semantic,instance_to_semantic_num = parse_superpoint_file(file_path)

# 打印统计结果
print_results(instance_to_semantic,instance_to_semantic_num)