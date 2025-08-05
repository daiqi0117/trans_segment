import os
import torch
import numpy as np
from torch_geometric.data import Data

# 超点文件和超点图文件的路径
superpoint_txt_dir = r'model/updated_superpoints/test/txt'
superpoint_pt_dir = r'model/updated_superpoints/test/pt'

# 输出路径（保存更新后的 .pt 文件，可以更新为不同目录）
output_pt_dir = r'model/updated_superpoints/test/updated_pts'
os.makedirs(output_pt_dir, exist_ok=True)  # 确保输出文件夹存在

# 遍历所有的 txt 文件
for txt_file in os.listdir(superpoint_txt_dir):
    if txt_file.endswith('.txt'):  # 确保只处理 .txt 文件
        txt_path = os.path.join(superpoint_txt_dir, txt_file)

        # 提取对应的 pt 文件名
        pt_file = txt_file.replace('.txt', '.pt')
        pt_path = os.path.join(superpoint_pt_dir, pt_file)

        # 检查对应的 pt 文件是否存在
        if not os.path.exists(pt_path):
            print(f"对应图文件 {pt_file} 不存在，跳过...")
            continue

        # 1. 读取超点文件，并提取后 17 维特征
        superpoint_data = np.loadtxt(txt_path)  # 加载超点的 txt 文件 (n x 20 矩阵)
        features = superpoint_data[:, 3:]      # 后 17 维特征，形状为 (n, 17)

        # 2. 读取对应的超点图 .pt 文件
        data = torch.load(pt_path)  # 加载超点图，应该是 torch_geometric.data.Data 格式

        # 确保节点数量一致（防止数据对应错误）
        if data.x.size(0) != features.shape[0]:
            print(f"超点文件 {txt_file} 和图文件 {pt_file} 的节点数量不匹配，跳过...")
            continue

        # 3. 更新超点图的节点特征
        data.x = torch.tensor(features, dtype=torch.float32)  # 将 numpy 转换为 PyTorch Tensor

        # 4. 保存更新后的图
        output_path = os.path.join(output_pt_dir, pt_file)
        torch.save(data, output_path)
        print(f"成功更新并保存 {output_path}")