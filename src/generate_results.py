import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

from .deepdepth.simi_clusters import partition_graph
from .compute_SDs import compute_edge_tuples


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

