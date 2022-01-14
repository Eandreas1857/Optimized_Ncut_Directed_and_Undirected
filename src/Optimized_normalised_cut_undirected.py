import scipy.linalg
import numpy as np
from numpy import linalg
import networkx as nx

def sym_weight_matrix(G, nodelist, data):
    """Returns the affinity matrix (usually denoted as A, but denoted as W here) 
    of the Graph G. W is an a asymmetric square matrix with non-negative real 
    numbers. W(i,j) represents the weight of the directed edge i --> j. [1]

    Parameters
    ----------
    G : NetworkX Graph

    nodelist : collection
        A collection of nodes in `G`. If not specified, nodes are all nodes in G
        in G.nodes() order.

    data : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    array
        The affinity matrix of G.
        
    References
    ----------
    .. [1] Jianbo Shi and Jitendra Malik.
           *Normalized Cuts and Image Segmentation*.
           <https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf>

    """
    w = nx.get_edge_attributes(G, data)

    nlen = len(nodelist)
    index = dict(zip(nodelist, range(nlen)))

    W = np.full((nlen, nlen), np.nan, order=None)
    for s in nodelist:
        for t in G.neighbors(s):
            if t in nodelist:
                try:
                    W[index[s], index[t]] = w[s,t]
                except:
                    try:
                        W[index[s], index[t]] = w[t,s]
                    except:
                        continue
                    continue

    W[np.isnan(W)] = 0
    W = np.asarray(W, dtype=None)
    return W

def sym_weighted_degree_matrix(G, nodelist, data):
    """Returns the degree D(i) of each node i in G, represented as a diagonal
    matrix D. D(i,j) = 0 if i != j, D(i,j) = D(i) otherwise. [1]

    Parameters
    ----------
    G : NetworkX Graph

    nodelist : collection
        A collection of nodes in `G`. If not specified, nodes are all nodes in G
        in G.nodes() order.

    data : object
        Edge attribute key to use as weight. If not specified, D(i) represents the 
        total number of edges from node i.

    Returns
    -------
    array
        The Degree matrix of G. 

    References
    ----------
    .. [1] Jianbo Shi and Jitendra Malik.
           *Normalized Cuts and Image Segmentation*.
           <https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf>

    """
    diag = []
    w = nx.get_edge_attributes(G, data)

    for node in nodelist:
        d = 0
        edges = [n for n in G.neighbors(node) if n in nodelist]
        for n in edges:
            try:
                d += w[n,node]
            except:
                try:
                    d += w[node,n]
                except:
                    continue
                continue
        diag.append(d)
    D = np.diag(diag)
    return D

def normalized_Laplacian(D, W):
    """Returns the normalized Laplacian nL. i.e., nL = D^(-1/2)(D-W)D^(-1/2) = 1-D^(-1/2)WD^(-1/2) [1].

    Parameters
    ----------
    D : array
        Degree diagonal matrix, see sym_weighted_degree_matrix.

    W : array
        Weight matrix, see sym_weighted_degree_matrix.
    
    Returns
    -------
    array
        The normalized Laplacian matrix.  

    References
    ----------
    .. [1] Jianbo Shi and Jitendra Malik.
           *Normalized Cuts and Image Segmentation*.
           <https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf>

    """
    L = np.subtract(D,W)
    D1 = scipy.linalg.fractional_matrix_power(D, -1/2)
    lmL = np.dot(D1,L)
    rmL = np.dot(lmL,D1)

    return rmL

def second_smallest_eigval_vec(nL):
    """Returns second smallest eigenvector of the normalized 
    Laplacian matrix nL, which is used to partition G into two
    clisters [1].

    Parameters
    ----------
    nL : array
        normalised Laplacian matrix, see normalized_Laplacian.
    
    Returns
    -------
    second_smallest_eigval_vec : array
        second smallest eigenvector of nL.

    References
    ----------
    .. [1] Jianbo Shi and Jitendra Malik.
           *Normalized Cuts and Image Segmentation*.
           <https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf>
    """
    eigval, eigvec = linalg.eig(nL)

    val = [round(i.real,4) for i in eigval]

    vec = []
    for i in range(len(eigvec)):
        v = [round(n.real,4) for n in eigvec[:,i]]
        vec.append(v)

    s = list(val).copy()
    s.sort()

    for i in range(len(eigval)):
        if val[i] == s[1]:
            second_smallest_eigval_vec = vec[i]
            break
 
    return second_smallest_eigval_vec

def partition_G(nodelist, second_smallest_eigval_vec):
    """Returns a partition of the graph G basied on second smallest eigenvector
    of the normalized Laplacian matrix nL.

    Parameters
    ----------
    nodelist : collection
        A collection of nodes in `G`. 
    
    second_smallest_eigval_vec : array 
        Second smallest eigenvector of nL, see second_smallest_eigval_vec.

    Returns
    -------
    A : list
        set of nodes in G in cluster 1.
    B : list
        set of nodes in G in cluster 2.

    References
    ----------
    .. [1] Jianbo Shi and Jitendra Malik.
           *Normalized Cuts and Image Segmentation*.
           <https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf>
    """
    A = []
    B = []
    for i in range(len(nodelist)):

        if second_smallest_eigval_vec[i] < 0:
            B.append(nodelist[i])
        if second_smallest_eigval_vec[i] > 0:
            A.append(nodelist[i])

    return A, B

def optimized_normalized_cut(G, nodelist = None, data = None, return_partition = True):
    """Returns second smallest eigenvector of the normalized Laplacian matrix nL. 
    If specified, also return the partition of the graph G based on second smallest eigenvector.

    Parameters
    ----------
    G : Networkx DiGraph

    nodelist : collection
        A collection of nodes in `G`. If not specified, nodes are all nodes in G
        in G.nodes() order.

    data : object
        Edge attribute key to use as weight. If not specified, D(i) represents the 
        total number of out edges from node i.    
    
    Returns
    -------
    ssV : array
        second smallest eigenvector of nL
    A : list
        set of nodes in G in cluster 1.
    B : list
        set of nodes in G in cluster 2.
    ONcut : number
        The optimised normalized cut of the partition, see [1] for more details.
    
    References
    ----------
    .. [1] Jianbo Shi and Jitendra Malik.
           *Normalized Cuts and Image Segmentation*.
           <https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf>
    """
    if nodelist == None:
        nodelist = list(G.nodes())

    W = sym_weight_matrix(G, nodelist, data)
    D = sym_weighted_degree_matrix(G, nodelist, data)

    nL = normalized_Laplacian(D, W)

    ssV = second_smallest_eigval_vec(nL)

    A, B = partition_G(nodelist, ssV)
    if A !=[]:
        ONcut = nx.algorithms.normalized_cut_size(G, A, weight = data)
    elif B != []:
        ONcut = nx.algorithms.normalized_cut_size(G, B, weight = data)

    return (ssV, A, B, ONcut) if return_partition == True else (ssV, ONcut)
