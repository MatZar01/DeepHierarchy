"""
@author: mromaszewski@iitis.pl, kfilus@iitis.pl
Similarity Matrix Clustering
"""

import numpy as np
from sklearn.cluster import OPTICS
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import HDBSCAN
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import community as community_louvain


def partition_graph(G, method="", **kwargs):
    """
    Partition the graph using the specified method.
    """
    assert method in ["louvain", "optics", "affinity", "hdbstan", "birch"]
    # G = nx.from_pandas_adjacency(similarity_matrix)
    if method == "louvain":
        return community_louvain.best_partition(G, **kwargs)

    similarity_matrix = nx.to_pandas_adjacency(G)
    distance_matrix = 1 - similarity_matrix

    if method == "optics":
        kwargs.setdefault("eps", 0.25)
        cls = OPTICS(metric="precomputed", **kwargs)
        m_to_use = distance_matrix
    elif method == "hdbstan":
        cls = HDBSCAN(metric="precomputed", **kwargs)
        m_to_use = distance_matrix
    elif method == "affinity":
        cls = AffinityPropagation(affinity="precomputed", **kwargs)
        m_to_use = similarity_matrix
    else:
        raise ValueError(f"Unknown method: {method}")

    labels = cls.fit_predict(m_to_use)
    partition = dict(zip(m_to_use.columns, labels))
    return partition