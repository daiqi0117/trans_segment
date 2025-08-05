import numpy as np
import os

# 加载点云文件
def load_point_cloud(file_path):
    """
    加载点云文件数据 (n*11的txt)
    """
    return np.loadtxt(file_path)


# 加载超点文件
def load_superpoint(file_path):
    """
    加载超点文件数据 (n*16的txt)
    """
    return np.loadtxt(file_path)


# 计算点云的全局错分概率
def calculate_global_misclassification_probabilities(point_cloud, superpoint_info):
    """
    参数：
        point_cloud: 原始点云数据 (n*11 numpy数组)
        superpoint_info: 超点数据 (n*16 numpy数组)
    返回：
        全局语义标签错分概率、全局实例标签错分概率
    """
    # 从原始点云中提取相关数据
    original_semantic_labels = point_cloud[:, 3]  # 原始语义标签（第4列）
    original_instance_labels = point_cloud[:, 4]  # 原始实例标签（第5列）
    point_superpoint_ids = point_cloud[:, 5]  # 所属超点序号（第6列）

    # 从超点文件中提取超点的语义和实例标签
    superpoint_labels_map = {}
    for row in superpoint_info:
        sp_id = int(row[0])  # 超点ID（第1列）
        semantic_label = int(row[1])  # 超点语义标签（第2列）
        instance_label = int(row[2])  # 超点实例标签（第3列）
        superpoint_labels_map[sp_id] = (semantic_label, instance_label)

        # 初始化错误计数
    total_points = point_cloud.shape[0]
    semantic_misclassified = 0
    instance_misclassified = 0

    # 遍历每个点，比较超点标签和原始标签是否一致
    for i in range(total_points):
        sp_id = int(point_superpoint_ids[i])  # 当前点所属的超点ID
        original_sem = int(original_semantic_labels[i])  # 当前点的原始语义标签
        original_inst = int(original_instance_labels[i])  # 当前点的原始实例标签

        # 获取超点的标签
        if sp_id in superpoint_labels_map:
            superpoint_sem, superpoint_inst = superpoint_labels_map[sp_id]

            # 检查语义标签是否一致
            if original_sem != superpoint_sem:
                semantic_misclassified += 1

                # 检查实例标签是否一致
            if original_inst != superpoint_inst:
                instance_misclassified += 1

                # 计算全局错分概率
    semantic_error_prob = semantic_misclassified / total_points
    instance_error_prob = instance_misclassified / total_points

    return semantic_error_prob, instance_error_prob


# 主函数
if __name__ == "__main__":
    data_root1 = 'output_train'
    data_root2 = 'updated_superpoints_updated'
    file_list1 = os.listdir(data_root1)
    file_list2 = os.listdir(data_root2)
    for file_path1,file_path2  in zip(file_list1,file_list2):
        # 示例文件路径
        point_cloud_file = data_root1+'/'+file_path1
        superpoint_file = data_root2+'/'+file_path2
        print(file_path1,file_path2)
        # 加载数据
        point_cloud = load_point_cloud(point_cloud_file)
        superpoint_info = load_superpoint(superpoint_file)

        # 计算全局错分概率
        global_semantic_error_prob, global_instance_error_prob = calculate_global_misclassification_probabilities(
            point_cloud, superpoint_info
        )

        # 输出全局错分概率
        print("Global Semantic Misclassification Probability: {:.4f}".format(global_semantic_error_prob))
        print("Global Instance Misclassification Probability: {:.4f}".format(global_instance_error_prob))
