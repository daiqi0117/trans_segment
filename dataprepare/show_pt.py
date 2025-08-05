import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# Load the graph file
graph = torch.load('model/updated_superpoints/train/updated_pts/2519_superpoint_50_10_50.pt')  # Replace with your actual file name

# Extract edge_index and node features
edge_index = graph.edge_index  # Shape: [2, num_edges]
node_features = graph.x  # Shape: [num_nodes, num_features]

# Count the number of edges for each node
num_nodes = node_features.size(0)
edge_counts = torch.bincount(edge_index[0], minlength=num_nodes)
print("Number of edges for each node:", edge_counts.tolist())

# Extract the first three dimensions of node features as coordinates
coordinates = node_features[:].numpy()
print(node_features.shape,coordinates)
# Convert the graph to a NetworkX graph for visualization
G = to_networkx(graph, to_undirected=True)

# Visualize the graph in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes
ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c='b', s=20, label='Nodes')

# Plot edges
for edge in G.edges():
    start, end = edge
    x = [coordinates[start, 0], coordinates[end, 0]]
    y = [coordinates[start, 1], coordinates[end, 1]]
    z = [coordinates[start, 2], coordinates[end, 2]]
    ax.plot(x, y, z, c='gray', alpha=0.5)

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.title('3D Visualization of Nodes and Edges')
plt.show()