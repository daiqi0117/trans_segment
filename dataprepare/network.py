import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv as SegaConv
from torch_geometric.nn import BatchNorm
import random
import time

# 确保设备为 GPU，如果不可用则降级为 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### 1. 数据加载和整合 ###
# 加载超点文件
def load_superpoint_features(file_path):
    return torch.tensor(np.loadtxt(file_path), dtype=torch.float32, device=device)  # 数据直接转 GPU


# 加载点云文件
def load_point_features(file_path):
    return torch.tensor(np.loadtxt(file_path), dtype=torch.float32, device=device)  # 数据直接转 GPU


# 加载图（torch_geometric 格式的 .pt 文件）
def load_graph(file_path):
    graph = torch.load(file_path)  # 加载图数据
    graph.x = graph.x.to(device) if graph.x is not None else None  # 节点特征转到 GPU
    graph.edge_index = graph.edge_index.to(device)  # 边索引转到 GPU
    return graph


### 2. 不同特征增强方案 ###
# 方案1：直接使用现有超点特征
def superpoint_feature_direct(superpoint_features):
    return superpoint_features.to(device)  # 确保张量位于 GPU


# 方案2：几何统计特征
def superpoint_feature_with_stats(superpoint_features, point_features):
    superpoint_ids = superpoint_features[:, 0].long()  # 超点索引
    num_superpoints = superpoint_ids.max() + 1
    extra_features = []

    for i in range(num_superpoints):
        mask = point_features[:, 5] == i
        points_in_superpoint = point_features[mask]
        avg_features = points_in_superpoint[:, 6:11].mean(dim=0)
        extra_features.append(avg_features)

    extra_features = torch.stack(extra_features).to(device)
    combined_features = torch.cat([superpoint_features, extra_features], dim=1)
    return combined_features


# 最远点采样 (FPS)
def farthest_point_sampling(points, n_samples):
    """
    最远点采样（Farthest Point Sampling, FPS）。
    从点云中采样固定数量的点，确保采样的点分布尽可能均匀。

    参数:
    - points: [n, d] 的 Tensor，表示输入点云，n 是点数量，d 是特征维度（仅支持 GPU 张量）。
    - n_samples: 希望采样的点数。

    返回:
    - Tensor: [n_samples, d] 的采样点集。
    """
    if points.size(0) < n_samples:  # 若点数不足，则直接补零点
        padded_points = torch.zeros((n_samples, points.size(1)), device=points.device)
        padded_points[:points.size(0), :] = points
        return padded_points

        # 初始化采样点
    sampled_idx = [random.randint(0, points.size(0) - 1)]  # 随机选择第一个点的索引
    sampled_points = points[sampled_idx]  # 将第一个点加入采样点集

    for _ in range(n_samples - 1):
        # 计算当前采样点到剩余点的最近距离
        dists = torch.cdist(points.unsqueeze(0), sampled_points.unsqueeze(0))[0]  # [n, sampled]
        min_dist, _ = dists.min(dim=1)  # [n]，每个点到采样点集的最小距离

        # 找到距离最远的点并加入采样点集
        farthest_idx = min_dist.argmax().item()
        sampled_idx.append(farthest_idx)
        sampled_points = torch.cat([sampled_points, points[farthest_idx].unsqueeze(0)], dim=0)

    return sampled_points

# 方案3：50fps叠加
def superpoint_feature_with_50FPS(superpoint_features, point_features):
    superpoint_ids = superpoint_features[:, 0].long()  # 超点索引
    num_superpoints = superpoint_ids.max() + 1
    extra_features = []

    for i in range(num_superpoints):
        mask = point_features[:, 5] == i
        points_in_superpoint = point_features[mask][:, :3]
        avg_features = farthest_point_sampling(points_in_superpoint, 50).flatten()

        extra_features.append(avg_features)

    extra_features = torch.stack(extra_features).to(device)
    combined_features = torch.cat([superpoint_features, extra_features], dim=1)
    print(combined_features.shape)
    return combined_features


### 3. 数据集构建 ###
def create_dataset(pointcloud_dir, superpoint_dir, graph_dir, feature_method='direct'):
    dataset = []
    pointcloud_files = sorted(os.listdir(pointcloud_dir))
    superpoint_files = sorted(os.listdir(superpoint_dir))
    graph_files = sorted(os.listdir(graph_dir))

    for p_file, sp_file, g_file in zip(pointcloud_files, superpoint_files, graph_files):
        point_features = load_point_features(os.path.join(pointcloud_dir, p_file))
        superpoint_features = load_superpoint_features(os.path.join(superpoint_dir, sp_file))
        graph_data = load_graph(os.path.join(graph_dir, g_file))

        if feature_method == 'direct':
            features = superpoint_feature_direct(superpoint_features)
        elif feature_method == 'stats':
            features = superpoint_feature_with_stats(superpoint_features, point_features)
        elif feature_method == 'fps':
            features = superpoint_feature_with_50FPS(superpoint_features, point_features)
        else:
            raise ValueError(f"Invalid feature_method: {feature_method}")

        graph_data.x = features
        dataset.append(graph_data)

    return dataset


### 4. 图神经网络 ###
class GraphSegmentationNet(nn.Module):
    def __init__(self, in_dim):
        super(GraphSegmentationNet, self).__init__()
        self.conv1 = SegaConv(in_dim, 128).to(device)
        self.bn1 = BatchNorm(128).to(device)
        self.conv2 = SegaConv(128, 256).to(device)
        self.bn2 = BatchNorm(256).to(device)
        self.conv3 = SegaConv(256, 512).to(device)
        self.bn3 = BatchNorm(512).to(device)
        self.conv4 = SegaConv(512, 512).to(device)
        self.bn4 = BatchNorm(512).to(device)
        self.conv5 = SegaConv(512, 512).to(device)
        self.bn5 = BatchNorm(512).to(device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        print(batch)
        x = F.leaky_relu(self.bn1(self.conv1(x, edge_index)))
        x = F.leaky_relu(self.bn2(self.conv2(x, edge_index)))
        x = F.leaky_relu(self.bn3(self.conv3(x, edge_index)))
        x = F.leaky_relu(self.bn4(self.conv4(x, edge_index)))
        x = self.bn5(self.conv5(x, edge_index))
        return x


### 5. 主程序 ###
if __name__ == "__main__":
    pointcloud_dir = "super_train_50_10_50/txt"  # 点云路径
    superpoint_dir = "super_train_50_10_50/superpoint"  # 超点路径
    graph_dir = "super_test_50_10_50/pt"  # 图路径

    start_time = time.time()
    print('开始')
    # 构建数据集
    dataset = create_dataset(pointcloud_dir, superpoint_dir, graph_dir, feature_method='fps')
    end_time = time.time()
    print(end_time-start_time)


    dataloader = DataLoader(dataset, batch_size=4)
    end_time = time.time()
    print(end_time-start_time)
    # 初始化模型
    model = GraphSegmentationNet(in_dim=dataset[0].x.shape[1])
    end_time = time.time()
    print(end_time-start_time)
    # 训练前向传播
    for batch in dataloader:
        batch = batch.to(device)  # 数据转 GPU
        output = model(batch)
        print("Output shape:", output.shape)
        end_time = time.time()
        print(end_time - start_time)