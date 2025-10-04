import networkx as nx
import random
import matplotlib.pyplot as plt
import os


def to_edge_list(edges, data=False):
    """
    :param edges: dict, key: edge name, value: edge attributes
    :param data: bool, whether to include edge attributes
    :return: list of tuple of str (src, tgt) where src/tgt are source/target node names
    """
    if type(edges) is dict:
        if data:
            return {(e["source"], e["target"]): name for name, e in edges.items()}
        else:
            return [(e["source"], e["target"]) for name, e in edges.items()]
    else:
        assert not data
        return [(e["source"], e["target"]) for e in edges]
    
    
def plot_nx_graph(G, outpath=None, show=True, edge_info=None, node_info=None, figsize=(12,12)):
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="")
    except: 
        print("graphviz not installed or error. Using spring layout.")
        pos = nx.spring_layout(G)
    pos = {k: (v[0], v[1] + (random.random() - 0.5) * 40) for k, v in pos.items()}
    fig = plt.figure(figsize=figsize)
    if edge_info:
        nx.draw_networkx_edge_labels(G, pos, edge_labels={k: edge_info[k] for k in G.edges}, font_color="red")
    if node_info:
        pos_ = {k: (v[0] + 5, v[1] + 5) for k, v in pos.items()}
        nx.draw_networkx_labels(G, pos_, labels=node_info, font_color="red", font_size=figsize[0], font_weight='bold')
    nx.draw(G, pos, with_labels=True)
    fig.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath)
    if show:
        plt.show()
    else:
        plt.close()


