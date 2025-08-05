import os
import numpy as np

# 定义文件夹路径
update_train_folder = 'update_test'
train_folder = 'super_test_50_10_50/txt'
output_train_folder = 'output_test'

# 创建输出目录（如果不存在）
if not os.path.exists(output_train_folder):
    os.makedirs(output_train_folder)

# 获取新点云和原始点云的文件列表
update_files = sorted([f for f in os.listdir(update_train_folder) if f.endswith('.txt')])
train_files = sorted([f for f in os.listdir(train_folder) if f.endswith('.txt')])

# 将文件名前四位作为键构建索引
update_dict = {f[:4]: f for f in update_files}  # 新点云文件名字典
train_dict = {f[:4]: f for f in train_files}    # 原始点云文件名字典

# 遍历原始点云文件
for key in train_dict:
    if key in update_dict:
        # 文件路径
        train_file_path = os.path.join(train_folder, train_dict[key])
        update_file_path = os.path.join(update_train_folder, update_dict[key])

        # 读取原始点云 (n×11 矩阵)
        train_data = np.loadtxt(train_file_path)
        # 读取新点云 (n×5 矩阵)
        update_data = np.loadtxt(update_file_path)

        # 替换原始点云的语义标签 (第4列)
        if train_data.shape[0] != update_data.shape[0]:
            print(f"文件 {train_dict[key]} 和 {update_dict[key]} 的点数不同，跳过！")
            continue
        train_data[:, 3] = update_data[:, 3]

        # 保存更新后的点云到输出文件夹
        output_file_path = os.path.join(output_train_folder, train_dict[key])
        np.savetxt(output_file_path, train_data)

        print(f"完成文件 {train_dict[key]} 的语义标签替换")
    else:
        print(f"没有找到匹配的新点云文件：{train_dict[key]}")

print("所有文件处理完成！")