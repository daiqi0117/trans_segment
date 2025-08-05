import os
import numpy as np

# 文件夹路径
folder_path = "updated_superpoints/train/txt"

# 遍历文件夹中的所有txt文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.txt'):  # 确保只处理txt文件
        file_path = os.path.join(folder_path, file_name)

        # 加载txt文件中的数据
        try:
            data = np.loadtxt(file_path)  # 使用numpy加载文件内容
            if data.shape[1] < 3:  # 如果矩阵的列数小于3，说明格式不对
                print(f"文件 {file_name} 格式错误，列数不足")
                continue

                # 提取第三列（超点的实例标签）并计算最大值
            instance_labels = data[:, 2]  # 第3列（索引从0开始）
            max_instance_label = int(np.max(instance_labels))  # 找到最大值

            print(f"文件 {file_name} 的实例标签最大值为: {max_instance_label}")

        except Exception as e:  # 捕获加载或处理数据时的异常
            print(f"处理文件 {file_name} 时出错：{str(e)}")