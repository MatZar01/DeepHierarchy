import matplotlib.pyplot as plt
import yaml
from nltk.corpus import wordnet as wn
import networkx as nx
from datetime import datetime
import pandas as pd
import numpy as np
import random
import os
import warnings

from .deepdepth.simi_clusters import partition_graph
from .deepdepth.reading_files import read_CSM, process_CSM
from .deepdepth.graph_tools import create_cluster_msts

from .print_style import Style


def wup_similarity_depth(one, other, verbose=False, simulate_root=True):
    """
    Based on wordnet Wu-Palmer Similarity from https://www.nltk.org/howto/wordnet.html
    Wu-Palmer Similarity:
    Return a score denoting how similar two word senses are, based on the
    depth of the two senses in the taxonomy and that of their Least Common
    Subsumer (most specific ancestor node). Previously, the scores computed
    by this implementation did _not_ always agree with those given by
    Pedersen's Perl implementation of WordNet Similarity. However, with
    the addition of the simulate_root flag (see below), the score for
    verbs now almost always agree but not always for nouns.

    The LCS does not necessarily feature in the shortest path connecting
    the two senses, as it is by definition the common ancestor deepest in
    the taxonomy, not closest to the two senses. Typically, however, it
    will so feature. Where multiple candidates for the LCS exist, that
    whose shortest path to the root node is the longest will be selected.
    Where the LCS has multiple paths to the root, the longer path is used
    for the purposes of the calculation.

    :type  other: Synset
    :param other: The ``Synset`` that this ``Synset`` is being compared to.
    :type simulate_root: bool
    :param simulate_root: The various verb taxonomies do not
        share a single root which disallows this metric from working for
        synsets that are not connected. This flag (True by default)
        creates a fake root that connects all the taxonomies. Set it
        to false to disable this behavior. For the noun taxonomy,
        there is usually a default root except for WordNet version 1.6.
        If you are using wordnet 1.6, a fake root will be added for nouns
        as well.
    :return: (depth,subsumer, similarity)
        - subsumer depth
        - subsumer
        - A float score denoting the similarity of the two ``Synset``
        objects, normally greater than zero. If no connecting path between
        the two senses can be found, None is returned.

    """
    need_root = one._needs_root() or other._needs_root()

    # Note that to preserve behavior from NLTK2 we set use_min_depth=True
    # It is possible that more accurate results could be obtained by
    # removing this setting and it should be tested later on
    subsumers = one.lowest_common_hypernyms(
        other, simulate_root=simulate_root and need_root, use_min_depth=True
    )

    # If no LCS was found return None
    if len(subsumers) == 0:
        return None

    subsumer = one if one in subsumers else subsumers[0]

    # Get the longest path from the LCS to the root,
    # including a correction:
    # - add one because the calculations include both the start and end
    #   nodes
    depth = subsumer.max_depth() + 1

    # Note: No need for an additional add-one correction for non-nouns
    # to account for an imaginary root node because that is now
    # automatically handled by simulate_root
    # if subsumer._pos != NOUN:
    #     depth += 1

    # Get the shortest path from the LCS to each of the synsets it is
    # subsuming.  Add this to the LCS path length to get the path
    # length from each synset to the root.
    len1 = one.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    len2 = other.shortest_path_distance(
        subsumer, simulate_root=simulate_root and need_root
    )
    if len1 is None or len2 is None:
        return None

    len1 += depth
    len2 += depth

    simi = (2.0 * depth) / (len1 + len2)
    return depth, subsumer, simi


def compute_wup(w1, w2):
    def get_synset_from_node_id(node_id: str):
        """
        Retrieve a WordNet synset using its node ID.
        :param node_id: WordNet node ID (e.g., "n01699831").
        :return: WordNet synset
        """
        pos = node_id[0]  # First character is the POS (n, v, a, r)
        offset = int(node_id[1:])  # Convert the remaining part to an integer
        return wn.synset_from_pos_and_offset(pos, offset)

    sa = get_synset_from_node_id(w1)

    sb = get_synset_from_node_id(w2)
    return wup_similarity_depth(sa, sb)


def list_nodes_edges(graph):
    """Returns lists of nodes and edges from a NetworkX graph."""
    nodes = list(graph.nodes())
    edges = list(graph.edges())
    return nodes, edges


def plot_graph(graph, save_flag=False, save_path=None, custom_labels=None):
    pos = nx.spring_layout(graph, weight="weight", iterations=100)
    edges, weights = zip(*nx.get_edge_attributes(graph, "weight").items())
    plt.figure(figsize=(16, 12), constrained_layout=False)
    # Draw the graph
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
    edge_colors = [cmap(norm(w)) for w in weights]

    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color='red',
        node_size=100,
        font_size=26,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("coolwarm"),
        width=2,
    )

    edge_labels = {(u, v): label for (u, v), label in zip(graph.edges(), custom_labels)}
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_color='black',
        font_size=26,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(weights)  # Pass edge weights for color scale
    cbar = plt.colorbar(sm)
    cbar.set_label('Edge Weight', fontsize=25)  # Set label with larger font size
    cbar.ax.tick_params(labelsize=25)

    if save_flag:
        current_time = datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S.%f')[:-3]
        saving_path = save_path + "\\" + formatted_time + ".png"
        plt.savefig(saving_path)
    plt.show()
    plt.close()


def compute_edge_tuples(graph):
    """
    Returns a DataFrame with columns: node1, node2, depth, subsumer, and simi
    for each edge in the graph.
    """
    data = []

    # Iterate over each edge in the graph
    for node1, node2 in graph.edges():
        # Compute the tuple for the edge nodes
        depth, subsumer, simi = compute_wup(node1, node2)

        # Append the results with separate elements of the tuple
        data.append((node1, node2, depth, subsumer, simi))

    # Create a DataFrame with expanded columns
    df = pd.DataFrame(data, columns=['node1', 'node2', 'depth', 'subsumer', 'simi'])
    return df


def compute_stats(similarity_matrix, classes, method_id=2, plot_flag=True, save_flag=False, save_path=None):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Function to retrieve the names corresponding to WordNet IDs
    def get_names_from_wnids(wnids):
        names = []
        for wnid in wnids:
            synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))  # Extract the numeric part and query
            names.append(synset.name().split('.')[0])  # Get the main name of the synset
        return names

    # Retrieve and print the names
    names = get_names_from_wnids(classes)

    similarity_matrix_no_diag = similarity_matrix.copy()
    np.fill_diagonal(similarity_matrix_no_diag, np.nan)

    # Flatten the matrix, excluding NaN (diagonal values)
    values = similarity_matrix_no_diag[~np.isnan(similarity_matrix_no_diag)].flatten()

    similarity_matrix = (similarity_matrix - np.min(similarity_matrix)) / (
                np.max(similarity_matrix) - np.min(similarity_matrix))
    similarity_matrix = pd.DataFrame(similarity_matrix, columns=classes)
    similarity_matrix.index = similarity_matrix.columns
    G = nx.from_pandas_adjacency(similarity_matrix)
    # Detect communities using different methods - grouping
    methods = ['louvain', 'optics', 'affinity', 'hdbstan']
    partition = partition_graph(G, method=methods[method_id])
    unique_values = set(partition.values())
    subgraphs = create_cluster_msts(G, partition)
    df_final = pd.DataFrame(columns=['node1', 'node2', 'depth', 'subsumer', 'simi'])
    for subgraph in subgraphs:
        df = compute_edge_tuples(subgraph)
        df_final = pd.concat([df_final, df], axis=0)
        if plot_flag:
            if len(subgraph) > 0:
                if not save_flag:
                    plot_graph(subgraph)
                else:
                    plot_graph(subgraph, save_flag=True, save_path=save_path)

    return df_final['depth'].mean(), df_final['simi'].mean(), len(unique_values)


def compute_depth_values(file_path, model_name, config):
    random.seed(config['SEED'])
    np.random.seed(config['SEED'])
    classes = load_classes(config)
    # Open the pickle file in read-binary mode and load its contents
    file_path_name = f'{file_path}/{model_name}/binary_matrices_after_epoch_weights_NCSM.pkl'
    similarity_matrix = read_CSM(file_path_name, CSM_index=-1)
    similarity_matrix = process_CSM(similarity_matrix, mode='NCSM')

    stats = []
    stats_groups = []
    for i in range(4):
        np.random.seed(42)
        stat = compute_stats(similarity_matrix, method_id=i, plot_flag=False, classes=classes)
        stats.append(stat[0])
        stats_groups.append(stat[2])

    return stats, stats_groups


def compute_net_depths(config):
    model_dir = config['MODEL_PATH']
    models = [f.path.split('/')[-1] for f in os.scandir(model_dir) if f.is_dir()]

    for model_name in models:
        np.random.seed(42)
        stats, stats_groups = compute_depth_values(model_dir, model_name, config)
        print(model_name + " & " + " & ".join(
            f"{round(num, 2):.2f} ({stats_groups[idx]})" for idx, num in enumerate(stats)))


def compute_wn_depths(config):
    random.seed(config['SEED'])
    np.random.seed(config['SEED'])
    classes = load_classes(config)
    # List all .pkl files in the folder
    folder_path = config['WN_PATH']
    pkl_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]

    # Open and load each .pkl file
    loaded_data = {}
    for file in pkl_files:
        file_path = os.path.join(folder_path, file)
        similarity_matrix = read_CSM(file_path, CSM_index=-1)
        similarity_matrix = process_CSM(similarity_matrix, mode='WCSM')
        stats = []
        stats_groups = []
        for i in range(4):
            stat = compute_stats(similarity_matrix, method_id=i, plot_flag=False, classes=classes)
            stats.append(stat[0])
            stats_groups.append(stat[2])
        print(f"WordNet {file.split('.')[0]} & " + " & ".join(
            f"{round(num, 2):.2f} ({stats_groups[idx]})" for idx, num in enumerate(stats)))


def load_classes(config):
    return yaml.load(open(f'{config["WN_PATH"]}/wordnet_classes.yml', 'r'), Loader=yaml.Loader)
