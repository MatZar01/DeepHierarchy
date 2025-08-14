"""
@author: mromaszewski@iitis.pl, kfilus@iitis.pl
@author: mzarski@iitis.pl -- getting this mess to actually work
Similarity Matrix Clustering
"""
import numpy as np
import networkx as nx


def is_similarity(matrix):
    """Check if a matrix looks like a similarity matrix."""
    # For a similarity matrix, diagonal often has largest values
    # compared to off-diagonal entries (self-similarity is highest).
    # Also check if values are within a plausible range [0, 1].
    diag_vals = np.diag(matrix)
    off_diag_vals = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    return diag_vals.mean().mean() >= off_diag_vals.mean().mean()


def create_cluster_msts(G, partition):
    """
    Creates Maximum Spanning Trees for each cluster on a similarity graph.
    """
    mst_subgraphs = []
    for cluster_id in set(partition.values()):
        nodes_in_cluster = [n for n, cid in partition.items() if cid == cluster_id]
        subgraph = G.subgraph(nodes_in_cluster)
        # Use maximum_spanning_tree for similarity-based MST
        mst = nx.maximum_spanning_tree(subgraph, weight="weight")
        mst_subgraphs.append(mst)
    return mst_subgraphs
