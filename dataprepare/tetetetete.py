import numpy as np


def read_txt_as_matrix(file_path):
    """
    读取txt文件并解析为numpy矩阵。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        data.append(values)
    return np.array(data)


def compute_box_vertices(center, lengths):
    """
    计算一个长方体的8个顶点。
    :param center: 长方体中心坐标 (x, y, z)。
    :param lengths: 长宽高的大小 (l, w, h)。
    :return: 8个顶点的坐标。
    """
    x, y, z = center
    l, w, h = lengths
    # 8个顶点相对中心点的偏移
    offsets = [
        [-l / 2, -w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [l / 2, w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
        [-l / 2, -w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [l / 2, w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
    ]
    vertices = [np.array(center) + np.array(offset) for offset in offsets]
    return vertices


def write_obj_file(output_path, all_vertices, all_faces):
    """
    将顶点和面写入.obj文件格式。
    :param output_path: 输出.obj文件路径。
    :param all_vertices: 所有顶点列表。
    :param all_faces: 所有面（顶点索引）列表。
    """
    with open(output_path, 'w') as f:
        # 写入顶点
        for v in all_vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            # 写入面
        for face in all_faces:
            f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")


def generate_obj_from_superpoints(input_file, output_file):
    """
    根据超点信息生成obj文件，同时输出长宽高中的最大值及对应的行。
    :param input_file: 输入的超点文本文件路径。
    :param output_file: 输出的.obj文件路径。
    """
    # 读取txt文件
    data = read_txt_as_matrix(input_file)

    # 定义存储顶点和面的列表
    all_vertices = []
    all_faces = []
    vertex_offset = 1  # 用于.obj格式中的顶点索引

    # 初始化最长长宽高及对应行
    max_length = 0  # 长的最大值
    max_width = 0  # 宽的最大值
    max_height = 0  # 高的最大值
    max_length_row = None  # 存储最长长对应的行
    max_width_row = None  # 存储最长宽对应的行
    max_height_row = None  # 存储最长高对应的行

    # 遍历每个超点
    for i, row in enumerate(data):
        center = row[3:6]  # 第4-6列是几何中心点(x, y, z)
        lengths = row[9:12]  # 第7-9列是长宽高(l, w, h)

        # 找出当前超点的长、宽、高
        l, w, h = lengths

        # 更新最长的长及对应的行
        if l > max_length:
            max_length = l
            max_length_row = row
            # 更新最长的宽及对应的行
        if w > max_width:
            max_width = w
            max_width_row = row
            # 更新最长的高及对应的行
        if h > max_height:
            max_height = h
            max_height_row = row

            # 计算当前超点长方体的8个顶点
        vertices = compute_box_vertices(center, lengths)

        # 添加这些顶点到all_vertices
        all_vertices.extend(vertices)

        # 定义这些顶点组成的面（使用.obj的风格，索引从1开始）
        faces = [
            [vertex_offset + 0, vertex_offset + 1, vertex_offset + 2, vertex_offset + 3],  # 底面
            [vertex_offset + 4, vertex_offset + 5, vertex_offset + 6, vertex_offset + 7],  # 顶面
            [vertex_offset + 0, vertex_offset + 1, vertex_offset + 5, vertex_offset + 4],  # 前面
            [vertex_offset + 2, vertex_offset + 3, vertex_offset + 7, vertex_offset + 6],  # 后面
            [vertex_offset + 1, vertex_offset + 2, vertex_offset + 6, vertex_offset + 5],  # 右面
            [vertex_offset + 0, vertex_offset + 3, vertex_offset + 7, vertex_offset + 4],  # 左面
        ]

        # 将面添加到all_faces
        all_faces.extend(faces)

        # 更新顶点偏移量
        vertex_offset += 8

        # 输出全局最长长宽高及对应行
    print(f"The longest length is: {max_length}, corresponding row: {max_length_row}")
    print(f"The longest width is: {max_width}, corresponding row: {max_width_row}")
    print(f"The longest height is: {max_height}, corresponding row: {max_height_row}")

    # 写入到.obj文件
    write_obj_file(output_file, all_vertices, all_faces)

# 示例使用
input_file = "updated_superpoints_updated/2519_superpoint_50_10_50.txt"  # 输入的超点文件
output_file = "superpoint_boxes.obj"  # 输出的obj文件
generate_obj_from_superpoints(input_file, output_file)

