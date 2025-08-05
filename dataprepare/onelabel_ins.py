import os
import numpy as np
from collections import Counter


def update_superpoint_labels(input_folder, output_folder):
    """
    更新超点文件中的语义标签，通过实例分组并选取每个实例中最常见的语义标签更新超点，
    并将结果保存到指定输出文件夹。

    参数:
        input_folder (str): 输入的超点文件夹路径
        output_folder (str): 输出更新后的超点文件夹路径
    """
    # 如果输出文件夹不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        # 获取输入文件夹的所有txt文件
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]

    for file in files:
        input_file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, file)

        # 读取文件内容
        data = np.loadtxt(input_file_path)  # 读取n*20矩阵

        # 提取需要使用的列
        superpoint_semantic_labels = data[:, 1]  # 第2列，原始语义标签
        superpoint_instance_labels = data[:, 2]  # 第3列，实例标签
        superpoint_point_counts = data[:, 10]  # 第11列，超点点的个数

        # 建立实例分组
        instance_dict = {}
        for i, instance_label in enumerate(superpoint_instance_labels):
            if instance_label not in instance_dict:
                instance_dict[instance_label] = []
                # 每个超点的信息 (语义标签, 点的个数)
            instance_dict[instance_label].append((superpoint_semantic_labels[i], superpoint_point_counts[i]))

            # 更新每个实例的语义标签
        instance_to_semantic = {}
        for instance_label, superpoints in instance_dict.items():
            # 计算每种语义标签的总点数
            label_counter = Counter()
            for semantic_label, point_count in superpoints:
                label_counter[semantic_label] += point_count
                # 选取点数最多的语义标签
            most_common_label = max(label_counter.items(), key=lambda x: x[1])[0]
            instance_to_semantic[instance_label] = most_common_label

            # 根据更新的实例语义标签更新数据
        for i, instance_label in enumerate(superpoint_instance_labels):
            data[i, 1] = instance_to_semantic[instance_label]  # 更新第2列语义标签

        # 保存更新后的数据到目标文件夹
        np.savetxt(output_file_path, data, fmt="%.6f")
        print(f"Updated and saved: {output_file_path}")

    # 使用函数更新语义标签


input_folder = "updated_superpoints/test/txt_orgin"  # 输入文件夹路径
output_folder = "updated_superpoints/test/txt"  # 输出文件夹路径
update_superpoint_labels(input_folder, output_folder)