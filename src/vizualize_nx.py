import networkx as nx
from IPython.display import Image, display
import random

def create_random_color_pallet(number_of_colors):
    pallet = []
    for j in range(number_of_colors+1):
        rand_colors = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
        pallet.append(rand_colors)
    return pallet

def view_pydot(G, data = None):
    pdot = nx.drawing.nx_pydot.to_pydot(G)
    if data != None:
        for i, edge in enumerate(pdot.get_edges()):
            edge.set_label(edge.get_attributes()[data])
    plt = Image(pdot.create_png())
    display(plt)

def view_clusters(G, clusters):
    pallet = create_random_color_pallet(len(clusters))
    color = {}

    for c in clusters:
        for i in G:
            if i in clusters[c]:
                color[i] = pallet[c][0]

    nx.set_node_attributes(G, color, 'color')

    view_pydot(G, 'weight')