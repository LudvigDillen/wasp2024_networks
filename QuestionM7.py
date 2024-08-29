import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import leidenalg as la
from sklearn.metrics import normalized_mutual_info_score

def generate_graphs(n_graphs, n_nodes, n_communities, p_in_values, p_out_values):
    graphs = []
    ground_truth = []
    
    # Generate ground truth community assignments
    sizes = [n_nodes // n_communities] * n_communities
    for community_id, size in enumerate(sizes):
        ground_truth.extend([community_id] * size)
    
    for p_in, p_out in zip(p_in_values, p_out_values):
        # Generate a graph using the Stochastic Block Model
        p_matrix = np.full((n_communities, n_communities), p_out)
        np.fill_diagonal(p_matrix, p_in)
        
        g = ig.Graph.SBM(n=n_nodes, pref_matrix=p_matrix.tolist(), block_sizes=sizes)
        
        graphs.append(g)
    
    return graphs, ground_truth

def analyze_partitions(graphs):
    partitions = []
    modularities = []
    for graph in graphs:
        partition = la.find_partition(graph, la.ModularityVertexPartition)
        partitions.append(partition)
        modularities.append(partition.modularity)
    
    return partitions, modularities

def visualize_graphs(graphs, partitions, modularities, ground_truth, p_in_values, p_out_values):
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'cyan']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    layout = graphs[0].layout("fr")  # Use the layout of the first graph for consistency
    layout_coords = {v.index: layout[v.index] for v in graphs[0].vs}  # Store layout coordinates
    
    # Plot the ground truth
    vertex_colors = [colors[community % len(colors)] for community in ground_truth]
    ig.plot(
        graphs[0], 
        target=axes[0], 
        vertex_color=vertex_colors,
        vertex_size=7,
        edge_width=0.5,
        layout=layout
    )
    axes[0].set_title("Ground Truth")
    
    # Plot the graphs with different p_in and p_out values
    for i, (graph, partition, modularity, p_in, p_out) in enumerate(zip(graphs, partitions, modularities, p_in_values, p_out_values), start=1):
        # Apply the same layout coordinates to each graph
        for v in graph.vs:
            v["x"], v["y"] = layout_coords[v.index]
        
        vertex_colors = [colors[community % len(colors)] for community in partition.membership]
        nmi_score = normalized_mutual_info_score(ground_truth, partition.membership)
        ig.plot(
            graph, 
            target=axes[i], 
            vertex_color=vertex_colors,
            vertex_size=7,
            edge_width=0.5,
            layout=layout
        )
        axes[i].set_title(f"p_in: {p_in:.2f}, p_out: {p_out:.2f}\nModularity: {modularity:.4f}\nNMI: {nmi_score:.4f}")
    
    plt.tight_layout()
    plt.savefig("graphs.pdf")
    plt.show()

# Example usage
n_graphs = 8
n_nodes = 100
n_communities = 5
p_in_values = np.linspace(0.1, .5, n_graphs)
p_out_values = np.linspace(0.01, 0.1, n_graphs)

graphs, ground_truth = generate_graphs(n_graphs, n_nodes, n_communities, p_in_values, p_out_values)
partitions, modularities = analyze_partitions(graphs)
visualize_graphs(graphs, partitions, modularities, ground_truth, p_in_values, p_out_values)