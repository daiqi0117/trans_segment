import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import torch

def load_point_cloud(file_path):
    """
    从txt文件加载点云。
    格式为 n*6: [x, y, z, semantic_label, instance_label, superpoint_label]
    """
    return np.loadtxt(file_path)


def build_point_edges_with_kdtree(points, k=16):
    """
    使用KDTree构建点之间的邻接关系 (k近邻)。

    points: ndarray[N, 3]，点的坐标 (x, y, z)
    k: 每个点查询 k 个近邻

    返回：
    edges: [(point_idx1, point_idx2), ...]，点与点的连接关系
    """
    tree = KDTree(points)
    edges = set()

    for i, point in enumerate(points):
        indices = tree.query(point, k=k + 1)[1]  # 查询 k+1 个邻居，包含自身
        for j in indices[1:]:  # 跳过自身
            if i < j:  # 去重，只保留 (i, j)，保证 i < j
                edges.add((i, j))
    return edges


def build_superpoint_graph_from_point_edges(edges, superpoint_labels):
    """
    通过点的邻接关系，构建超点图，仅表示连接关系，不添加边的其他属性。

    edges: [(point_idx1, point_idx2), ...]，点之间的连接关系
    superpoint_labels: ndarray[N]，每个点的超点标签

    返回：
    superpoint_edges: [(superpoint_id1, superpoint_id2), ...]，超点之间的边，仅表示是否连接
    """
    superpoint_edges = set()

    for i, j in edges:
        sp1 = superpoint_labels[i]
        sp2 = superpoint_labels[j]

        if sp1 != sp2:  # 仅记录跨超点的边
            superpoint_edges.add((min(sp1, sp2), max(sp1, sp2)))  # 去重，存储 (sp1, sp2)，保证 sp1 < sp2

    return list(superpoint_edges)


def compute_superpoint_centers(points, superpoint_labels):
    """
    计算每个超点的中心点，作为节点的特征。

    points: ndarray[N, 3]，点的坐标 (x, y, z)
    superpoint_labels: ndarray[N]，每个点的超点标签

    返回：
    superpoint_centers: 超点中心点坐标
    """
    superpoint_dict = defaultdict(list)

    for point, label in zip(points, superpoint_labels):
        superpoint_dict[label].append(point)

    superpoint_centers = []
    for label, pts in superpoint_dict.items():
        pts = np.array(pts)
        center = np.mean(pts, axis=0)  # 坐标均值作为中心点
        superpoint_centers.append(center)

    superpoint_centers = np.array(superpoint_centers)
    return superpoint_centers


def analyze_superpoint_instances(points, superpoint_labels, instance_labels, output_file=None):
    """
    分析每个超点包含的实例种类和每类点数量。

    points: ndarray[N, 3]，点的坐标 (x, y, z)
    superpoint_labels: ndarray[N]，每个点的超点标签
    instance_labels: ndarray[N]，每个点的实例标签
    output_file: str，可选，输出文件路径

    输出：
    每个超点包含的实例种类及其点的数量
    """
    superpoint_instance_info = defaultdict(lambda: defaultdict(int))

    for sp_label, instance_label in zip(superpoint_labels, instance_labels):
        superpoint_instance_info[sp_label][instance_label] += 1

        # 输出结果到文件或打印
    if output_file:
        with open(output_file, "w") as f:
            f.write("SuperpointID\tInstanceID\tPointCount\n")
            for sp_id, instances in superpoint_instance_info.items():
                for instance_id, count in instances.items():
                    f.write(f"{sp_id}\t{instance_id}\t{count}\n")
        print(f"实例统计已保存到 {output_file}")
    else:
        # 打印统计结果
        for sp_id, instances in superpoint_instance_info.items():
            print(f"Superpoint {sp_id}:")
            for instance_id, count in instances.items():
                print(f"  Instance {instance_id}: {count} points")

    return superpoint_instance_info


def save_superpoint_graph_torch_geometric(file_name, superpoint_centers, superpoint_edges):
    """
    使用torch_geometric保存超点图结构。

    file_name: 保存路径
    superpoint_centers: 超点的中心点 (节点特征)
    superpoint_edges: 超点之间的边连接
    """


    # 节点特征：中心点 (用超点的坐标表示)
    x = torch.tensor(superpoint_centers, dtype=torch.float)

    # 边索引：src 和 dst (转为 PyG 的 edge_index 格式)
    edge_index = torch.tensor(superpoint_edges, dtype=torch.long).T

    # 创建超点图
    data = Data(x=x, edge_index=edge_index)
    torch.save(data, file_name)

    print(f"超点图已保存为 Torch Geometric 格式到 {file_name}")


def visualize_superpoint_graph_plotly(superpoint_centers, superpoint_edges):
    """
    使用Plotly可视化超点图。

    superpoint_centers: 超点的中心点坐标
    superpoint_edges: 超点的边 (src, dst)
    """
    # 获取超点数量
    num_superpoints = len(superpoint_centers)

    # 为每个超点生成唯一颜色（颜色映射）
    colors = ["rgb({}, {}, {})".format(
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ) for _ in range(num_superpoints)]

    # 转换超点中心点的坐标
    x_coords = superpoint_centers[:, 0]
    y_coords = superpoint_centers[:, 1]
    z_coords = superpoint_centers[:, 2]

    # 创建节点的 3D 散点图
    scatter = go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=4,  # 节点大小
            color=colors,  # 节点颜色
            opacity=0.8
        ),
        name='Superpoint Nodes'
    )

    # 创建边的线段
    edge_lines = []
    for edge in superpoint_edges:
        src, dst = edge
        edge_lines.append(
            go.Scatter3d(
                x=[superpoint_centers[src, 0], superpoint_centers[dst, 0], None],  # x 坐标 (None 表示断开线段)
                y=[superpoint_centers[src, 1], superpoint_centers[dst, 1], None],  # y 坐标
                z=[superpoint_centers[src, 2], superpoint_centers[dst, 2], None],  # z 坐标
                mode='lines',
                line=dict(
                    color='rgba(200, 200, 200, 0.5)',  # 边的颜色
                    width=1  # 边宽度
                ),
                hoverinfo='none',  # 不显示边的 hover 信息
                showlegend=False  # 不显示边图例
            )
        )

        # 将边和节点添加到场景中
    fig = go.Figure(data=[scatter] + edge_lines)

    # 更新布局
    fig.update_layout(
        title="Superpoint Graph Visualization",
        scene=dict(
            xaxis_title='X Axis',
            yaxis_title='Y Axis',
            zaxis_title='Z Axis'
        ),
        showlegend=True
    )

    fig.write_html("superpoint_graph.html")
    # 显示图
    # fig.show()


def main(point_cloud_file, graph_file, instance_analysis_file, k=16):
    """
    主流程：从点云生成超点图，用 Torch Geometric 保存，并可视化图结构。
    同时对每个超点的实例种类及点数量进行统计。
    """

    # 加载点云
    data = load_point_cloud(point_cloud_file)
    points = data[:, :3]  # 提取 xyz 坐标
    semantic_labels = data[:, 3].astype(int)  # 提取语义标签
    instance_labels = data[:, 4].astype(int)  # 提取实例标签
    superpoint_labels = data[:, 5].astype(int)  # 超点标签

    # 构建点之间的邻接关系
    print("构建点之间的邻接关系...")
    point_edges = build_point_edges_with_kdtree(points, k=k)
    print(f"点邻接关系提取完成，共 {len(point_edges)} 条边")

    # 构建超点之间的连接关系
    print("构建超点之间的连接...")
    superpoint_edges = build_superpoint_graph_from_point_edges(point_edges, superpoint_labels)
    print(f"超点之间的边提取完成，共 {len(superpoint_edges)} 条边")

    # 计算超点中心点
    print("计算超点的中心点...")
    superpoint_centers = compute_superpoint_centers(points, superpoint_labels)

    # 保存超点图为 Torch Geometric 格式
    save_superpoint_graph_torch_geometric(graph_file, superpoint_centers, superpoint_edges)

    # 分析每个超点的实例种类和点数量
    print("分析每个超点的实例种类和点数量...")
    analyze_superpoint_instances(points, superpoint_labels, instance_labels, output_file=instance_analysis_file)

    # 可视化超点图
    print("可视化超点图...")
    visualize_superpoint_graph_plotly(superpoint_centers, superpoint_edges)


# 示例运行调用：
if __name__ == "__main__":
    main("0070_10_100.txt", "superpoint_graph.pt", "superpoint_instance_analysis.txt", k=16)