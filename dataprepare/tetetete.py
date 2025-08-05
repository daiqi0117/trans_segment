import torch

# 加载图数据
from torch_geometric.data import Data

# 加载两个超点图
graph1 = torch.load("super_train_50_10_50/updated_pt/0040_superpoint_50_10_50.pt")
graph2 = torch.load("temp/superpoint/0040_superpoint_50_10_50.pt")

# 比较两个图的边连接信息
def compare_edge_indices(graph1, graph2):
    # 检查 edge_index 是否存在
    if not hasattr(graph1, 'edge_index') or not hasattr(graph2, 'edge_index'):
        print("至少有一个图缺少 edge_index 信息")
        return False

    # 确保两个图的形状相同
    if graph1.edge_index.shape != graph2.edge_index.shape:
        print("两个图的边连接数量不同")
        return False

    # 比较每一条边（edge_index），但要确保顺序不影响结果
    # 由于边的顺序可能不同，可以对边的连接进行排序后再比较
    edge_index1 = graph1.edge_index.cpu().numpy()
    edge_index2 = graph2.edge_index.cpu().numpy()

    edge_index1_sorted = edge_index1.T.tolist()
    edge_index2_sorted = edge_index2.T.tolist()

    # 排序后再比较
    edge_index1_sorted.sort()
    edge_index2_sorted.sort()

    if edge_index1_sorted == edge_index2_sorted:
        print("两个图的边连接完全相同")
        return True
    else:
        print("两个图的边连接不同")
        return False

# 执行比较
compare_edge_indices(graph1, graph2)