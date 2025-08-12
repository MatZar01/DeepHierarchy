#%%
import pandas as pd
import sys

import pandas as pd
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import random

import pickle

from nltk.corpus import wordnet as wn
from deepdepth.simi_clusters import partition_graph
from deepdepth.reading_files import read_CSM, process_CSM
import networkx as nx
from datetime import datetime

from deepdepth.graph_tools import create_cluster_msts


#  -------------- wup_depth implementation ----------------
#  see https://www.nltk.org/howto/wordnet.html

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


def plot_graph(graph, save_flag=False, save_path=None, custom_labels=None, custom_nodes=None):
    pos = nx.spring_layout(graph, weight="weight", iterations=100)
    edges, weights = zip(*nx.get_edge_attributes(graph, "weight").items())
    plt.figure(figsize=(16, 12), constrained_layout=False)
    # Draw the graph
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
    edge_colors = [cmap(norm(w)) for w in weights]

    # node_labels = custom_nodes if custom_nodes else {node: node for node in graph.nodes()}
    # Ensure custom_labels is properly formatted
    if isinstance(custom_nodes, list):
        node_labels = {node: label for node, label in zip(graph.nodes(), custom_nodes)}
    elif isinstance(custom_nodes, dict):
        node_labels = {node: custom_nodes.get(node, node) for node in graph.nodes()}
    else:
        node_labels = {node: node for node in graph.nodes()}  # Default to original names

    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_color='red',
        node_size=100,
        font_size=26,
        edge_color=edge_colors,
        edge_cmap=plt.get_cmap("coolwarm"),
        width=2,
    )

    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=20, font_color='black')

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
#%%
def compute_stats(similarity_matrix, WN_ID_classes, method_id=2, save_path=None):
    import nltk
    from nltk.corpus import wordnet as wn
    import pandas as pd
    from deepdepth.graph_tools import create_cluster_msts
    import networkx as nx
    import community as community_louvain

    # Function to retrieve the names corresponding to WordNet IDs
    def get_names_from_wnids(wnids):
        names = []
        for wnid in wnids:
            synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))  # Extract the numeric part and query
            names.append(synset.name().split('.')[0])  # Get the main name of the synset
        return names

    # class names
    names = get_names_from_wnids(WN_ID_classes)

    similarity_matrix = pd.DataFrame(similarity_matrix, columns=WN_ID_classes)
    similarity_matrix.index = similarity_matrix.columns
    G = nx.from_pandas_adjacency(similarity_matrix)

    # Detect communities using different methods - grouping
    methods = ['louvain', 'optics', 'affinity', 'hdbstan']
    partition = partition_graph(G, method=methods[method_id])
    unique_values = set(partition.values())
    subgraphs = create_cluster_msts(G, partition)

    df_final = pd.DataFrame(columns=['node1', 'node2', 'depth', 'subsumer', 'simi'])

    for idx, subgraph in enumerate(subgraphs):
        df = compute_edge_tuples(subgraph)
        df['ID'] = idx
        df_final = pd.concat([df_final, df], axis=0)
    if save_path:
        df_final.to_csv(save_path)

    return df_final['depth'].mean(), df_final['simi'].mean(), len(unique_values)


def plot_subgraphs(similarity_matrix, WN_ID_classes, method_id=2, save_flag=False, save_path=None):
    import nltk
    from nltk.corpus import wordnet as wn
    import pandas as pd
    from deepdepth.graph_tools import create_cluster_msts
    import networkx as nx
    import community as community_louvain

    # Function to retrieve the names corresponding to WordNet IDs
    def get_names_from_wnids(wnids):
        names = []
        for wnid in wnids:
            synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))  # Extract the numeric part and query
            names.append(synset.name().split('.')[0])  # Get the main name of the synset
        return names

    # class names
    names = get_names_from_wnids(WN_ID_classes)

    mapper = dict(zip(WN_ID_classes, names))

    similarity_matrix = pd.DataFrame(similarity_matrix, columns=WN_ID_classes)
    similarity_matrix.index = similarity_matrix.columns
    G = nx.from_pandas_adjacency(similarity_matrix)

    # Detect communities using different methods - grouping
    methods = ['louvain', 'optics', 'affinity', 'hdbstan']
    partition = partition_graph(G, method=methods[method_id])
    unique_values = set(partition.values())
    subgraphs = create_cluster_msts(G, partition)

    df_final = pd.DataFrame(columns=['node1', 'node2', 'depth', 'subsumer', 'simi'])
    for idx, subgraph in enumerate(subgraphs):
        df = compute_edge_tuples(subgraph)
        df['ID'] = idx
        df_final = pd.concat([df_final, df], axis=0)
        names_list = [mapper[name] for name in subgraph.nodes()]
        if len(subgraph) > 0:
            if not save_flag:
                plot_graph(subgraph, custom_labels=df['depth'], custom_nodes=names_list)
            else:
                plot_graph(subgraph, save_flag=True, save_path=save_path, custom_labels=df['depth'],
                           custom_nodes=names_list)
#%%
WN_classes = ['n01532829', 'n01558993', 'n01704323', 'n01749939', 'n01770081', 'n01843383', 'n01855672', 'n01910747', 'n01930112', 'n01981276', 'n02074367', 'n02089867', 'n02091244', 'n02091831', 'n02099601', 'n02101006', 'n02105505', 'n02108089', 'n02108551', 'n02108915', 'n02110063', 'n02110341', 'n02111277', 'n02113712', 'n02114548', 'n02116738', 'n02120079', 'n02129165', 'n02138441', 'n02165456', 'n02174001', 'n02219486', 'n02443484', 'n02457408', 'n02606052', 'n02687172', 'n02747177', 'n02795169', 'n02823428', 'n02871525', 'n02950826', 'n02966193', 'n02971356', 'n02981792', 'n03017168', 'n03047690', 'n03062245', 'n03075370', 'n03127925', 'n03146219', 'n03207743', 'n03220513', 'n03272010', 'n03337140', 'n03347037', 'n03400231', 'n03417042', 'n03476684', 'n03527444', 'n03535780', 'n03544143', 'n03584254', 'n03676483', 'n03770439', 'n03773504', 'n03775546', 'n03838899', 'n03854065', 'n03888605', 'n03908618', 'n03924679', 'n03980874', 'n03998194', 'n04067472', 'n04146614', 'n04149813', 'n04243546', 'n04251144', 'n04258138', 'n04275548', 'n04296562', 'n04389033', 'n04418357', 'n04435653', 'n04443257', 'n04509417', 'n04515003', 'n04522168', 'n04596742', 'n04604644', 'n04612504', 'n06794110', 'n07584110', 'n07613480', 'n07697537', 'n07747607', 'n09246464', 'n09256479', 'n13054560', 'n13133613']

NCSM_depth = []
epoch = 200
for model in ['ConvNeXt', 'ViTB']:
    NCSM_file = f"Networks_CSM/{model}/binary_matrices_after_epoch_weights_NCSM.pkl"
    NCSM_df_save = f"RESULTS_ROOT/DF/NCSM/{epoch}/{model}/results.csv"
    NCSM_plot_save = f"RESULTS_ROOT/PLOTS/NCSM/{epoch}/{model}"

    similarity_matrix = read_CSM(NCSM_file, CSM_index=epoch)
    similarity_matrix = process_CSM(similarity_matrix, mode='NCSM')

    print(compute_stats(similarity_matrix, WN_ID_classes=WN_classes, method_id=2, save_path=NCSM_df_save))
    plot_subgraphs(similarity_matrix, WN_ID_classes=WN_classes, method_id=2, save_flag=True, save_path=NCSM_plot_save)
    sd, _, _ = compute_stats(similarity_matrix, WN_ID_classes=WN_classes, method_id=2)
    NCSM_depth.append(sd)
print(NCSM_depth)
data = {
    'NCSM_depth': NCSM_depth
}
df = pd.DataFrame(data)
#%%
CCSM_depth = []
epoch = 200
for model in ['ConvNeXt', 'ViTB']:
    CCSM_file = f"Networks_CSM/{model}/binary_matrices_after_epoch_confusion_CCSM.pkl"
    CCSM_df_save = f"RESULTS_ROOT/DF/CCSM/{epoch}/{model}/results.csv"
    CCSM_plot_save = f"RESULTS_ROOT/PLOTS/CCSM/{epoch}/{model}"

    similarity_matrix = read_CSM(CCSM_file, CSM_index=epoch)
    similarity_matrix = process_CSM(similarity_matrix, mode='CCSM')

    print(compute_stats(similarity_matrix, WN_ID_classes=WN_classes, method_id=2, save_path=CCSM_df_save))
    plot_subgraphs(similarity_matrix, WN_ID_classes=WN_classes, method_id=2, save_flag=True, save_path=CCSM_plot_save)
    sd, _, _ = compute_stats(similarity_matrix, WN_ID_classes=WN_classes, method_id=2)
    CCSM_depth.append(sd)
print(CCSM_depth)
df['CCSM_depth'] = CCSM_depth
#%%
import yaml
accs = []
losses = []
epoch = 200
for model in ['ConvNeXt', 'ViTB']:
    yml_file = f"/home/mateusz/Desktop/DeepHierarchyTemp/Networks_CSM/{model}/results.yml"
    with open(yml_file, 'r') as file:
        data = yaml.safe_load(file)
        accs.append(data['test']['acc'][epoch])
        losses.append(data['test']['loss'][epoch])
print(accs)
print(losses)
df["accuracy"] = accs
df["losses"] = losses
#%%
df.to_csv("RESULTS_ROOT/results_epoch_200.csv", index=False)