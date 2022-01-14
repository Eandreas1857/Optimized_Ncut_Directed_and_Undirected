import networkx as nx
from IPython.display import Image, display

def view_pydot(G, data = None):
    pdot = nx.drawing.nx_pydot.to_pydot(G)
    if data != None:
        for i, edge in enumerate(pdot.get_edges()):
            edge.set_label(edge.get_attributes()[data])
    plt = Image(pdot.create_png())
    display(plt)