import os
import numpy as np

# 定义文件夹路径
folder_path = "model/updated_superpoints_updated_test"

# 创建一个字典，用来存储超点长度 n 和对应文件名的映射
length_to_files = {}

# 遍历文件夹中的所有txt文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):  # 检查是否是txt文件
        file_path = os.path.join(folder_path, file_name)

        # 读取文件，加载超点数组
        try:
            array = np.loadtxt(file_path)  # 加载文件为numpy数组
            if array.ndim > 1:  # 确保是n*20的形状
                n = array.shape[0]  # 获取数组的行数 n
            else:
                # 文件中内容不符合 n*20 的形式，跳过处理
                print(f"警告：文件 {file_name} 不是预期的 n*20 格式，跳过。")
                continue

                # 将文件名添加到对应长度的字典中
            if n not in length_to_files:
                length_to_files[n] = []  # 如果 n 不在字典中，初始化一个空列表
            length_to_files[n].append(file_name)  # 将文件名追加到对应的长度条目

        except Exception as e:
            print(f"错误：无法读取文件 {file_name}，错误信息：{e}")
            continue

        # 输出字典为字符串，方便直接用作代码输入
print("超点长度和文件映射字典：")
print(length_to_files)