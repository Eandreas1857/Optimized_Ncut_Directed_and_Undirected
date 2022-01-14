import networkx as nx
import scipy.linalg
import numpy as np
from numpy import linalg

def asym_weight_matrix(G, nodelist = None, data = None):
    """Returns the affinity matrix (usually denoted as A, but denoted as W here) 
    of the DiGraph G. W is an a asymmetric square matrix with non-negative real 
    numbers. W(i,j) represents the weight of the directed edge i --> j. [1]

    Parameters
    ----------
    G : NetworkX DiGraph

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
    .. [1] Marina Meila and William Pentney.
           *Clustering by weighted cuts in directed graphs*.
           <https://sites.stat.washington.edu/mmp/Papers/sdm-wcuts.pdf>

    """
    if nodelist == None:
        nodelist = G.nodes()

    if data != None:   
        weight = nx.get_edge_attributes(G, data)

    nlen = len(nodelist)
    index = dict(zip(nodelist, range(nlen)))

    W = np.full((nlen, nlen), np.nan, order=None)
    for i in nodelist:
        for j in G.neighbors(i):
            if j in nodelist:
                W[index[i], index[j]] = 1 if data == None else weight[i,j]

    W[np.isnan(W)] = 0
    W = np.asarray(W, dtype=None)
    
    return W

def asym_weighted_degree_matrix(G, nodelist = None, data = None):
    """Returns the out-degree D(i) of each node i in G, represented as a diagonal
    matrix D. D(i,j) = 0 if i != j, D(i,j) = D(i) otherwise. [1]

    Parameters
    ----------
    G : NetworkX DiGraph

    nodelist : collection
        A collection of nodes in `G`. If not specified, nodes are all nodes in G
        in G.nodes() order.

    data : object
        Edge attribute key to use as weight. If not specified, D(i) represents the 
        total number of out edges from node i.

    Returns
    -------
    array
        The Out-Degree matrix of G.

    Notes:
    ------
    D(i) is said to be 1 if the node i has no out edges, this avoids divison
    by zero in further functions and makes node i a 'sink'. [1]   

    References
    ----------
    .. [1] Marina Meila and William Pentney.
           *Clustering by weighted cuts in directed graphs*.
           <https://sites.stat.washington.edu/mmp/Papers/sdm-wcuts.pdf>

    """
    if nodelist == None:
        nodelist = G.nodes()

    if data != None:   
        weight = nx.get_edge_attributes(G, data)

    diag = []

    for i in nodelist:
        
        edges = [n for n in G.neighbors(i) if n in nodelist]
        if edges == []:
            d = 1 # avoids divison by 0 and makes nodes with no out edges a sink.
            diag.append(d)
        else:
            d = 0
            for j in edges:
                d += 1 if data == None else weight[i, j]
            diag.append(d)
    D = np.diag(diag)
    return D

def Hermitian_normalized_Laplacian(D, W, T):
    """Returns the Hermitian part of nL, where nL is the normalized Laplacian. i.e., 
    for T = D, nL = D^(-1/2)(D-W)D^(-1/2) = 1-D^(-1/2)WD^(-1/2) [1].

    Parameters
    ----------
    D : array
        Out-Degree diagonal matrix, see asym_weighted_degree_matrix.

    W : array
        Weight matrix, see asym_weighted_degree_matrix.
    
    T : array
        User defined weighting of the nodes. For WNcut introduced in [1], use T = D.

    Returns
    -------
    array
        The Hermitian part of normalized Laplacian matrix.

    Notes:
    ------
    Given any matrix B, the Hermitian part of B is H(B) = (1/2)(B + B^T) and is always
    a symmetric matrix [1]. Further, [1] also introduces a row weight matrix T', though
    for the WNcut algorthm it is assumed that T'=1 so it omitted from this function. If 
    this is not the case, replace W by T'W and D by T'D [1].   

    References
    ----------
    .. [1] Marina Meila and William Pentney.
           *Clustering by weighted cuts in directed graphs*.
           <https://sites.stat.washington.edu/mmp/Papers/sdm-wcuts.pdf>

    """
    H = 2*D-W-W.transpose()
    T1 = scipy.linalg.fractional_matrix_power(T, -1/2)
    lmL = np.dot(T1,H)
    rmL = np.dot(lmL,T1)
    L = (1/2)*rmL
    return L

def k_smallest_eigvec(nodelist, L, T, k):
    """Returns the k smallest eigenvectors of the Hermitian part of the normalised
     Laplacian nL. 

    Parameters
    ----------
    G : NetworkX DiGraph

    L : array
        Hermitian part of normalized Laplacian matrix, see Hermitian_normalised_Laplacian
    
    T : array
        User defined weighting of the nodes. For WNcut introduced in [1], use T = D.

    k : int
        k is the number of clusters trying to find. Though it is suggested to use k = 2,
        and used the second smallest eigenvector to break up the graph into two groups
        then run the algorthm again on the groups to further break up the clusters [2].

    Returns
    -------
    array
        Matrix X = T^(-1/2)Y, where Y has k smallest eigenvectors of nL as a column vectors.

    Notes:
    ------
    The columns of Y are automatically orthogonal, since nL is real-valued, 
    symmetric matric.  

    References
    ----------
    .. [1] Marina Meila and William Pentney.
           *Clustering by weighted cuts in directed graphs*.
           <https://sites.stat.washington.edu/mmp/Papers/sdm-wcuts.pdf>

    .. [2] Jianbo Shi and Jitendra Malik.
           *Normalized Cuts and Image Segmentation*.
           <https://www.cis.upenn.edu/~jshi/papers/pami_ncut.pdf>

    """
    nlen = len(nodelist)
    eigval, eigvec = linalg.eig(L)
    T1 = scipy.linalg.fractional_matrix_power(T, -1/2)
    
    Y = np.full((nlen, k), np.nan, order=None)

    val = [round(i.real,4) for i in eigval]

    s = list(val).copy()
    s.sort()

    for y in range(len(s[:k])):
        for i in range(len(eigval)):
            if val[i] == s[:k][y]:
                n = linalg.norm(eigvec[:,i].transpose())
                Y[:,y] = eigvec[:,i]/n
                break
 
    X = np.dot(T1,Y)
    return X

def partition_G(nodelist, second_smallest_eigval_vec):
    """Returns a partition of the graph G basied on second smallest eigenvector
    of the Hermitian part of normalized Laplacian matrix nL.

    Parameters
    ----------
    nodelist : collection
        A collection of nodes in `G`. 
    
    second_smallest_eigval_vec : array 
        Second smallest eigenvector of nL, see k_smallest_eigvec.

    Returns
    -------
    A : list
        set of nodes in G in cluster 1.
    B : list
        set of nodes in G in cluster 2.

    """
    A = []
    B = []
    for i in range(len(nodelist)):
        if second_smallest_eigval_vec[i] > 0:
                    A.append(nodelist[i])
        if second_smallest_eigval_vec[i] < 0:
            B.append(nodelist[i])

    return A, B

def asym_optimized_normalized_cut(G, nodelist = None, data = None, return_partition = True):
    """Returns second smallest eigenvector of the Hermitian part of normalized 
    Laplacian matrix nL. If specified, also return the partition of the graph G 
    based on second smallest eigenvector.

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
    X[:,1] : array
        second smallest eigenvector of nL
    A : list
        set of nodes in G in cluster 1.
    B : list
        set of nodes in G in cluster 2.

    """
    if nodelist == None:
        nodelist = G.nodes()
    W = asym_weight_matrix(G, nodelist, data)
    D = asym_weighted_degree_matrix(G, nodelist, data)
    L = Hermitian_normalized_Laplacian(D, W, D)
    X = k_smallest_eigvec(nodelist, L, D, 2)

    if return_partition == True:
        A, B  = partition_G(nodelist, X[:,1])

    return (X[:,1], A, B) if return_partition == True else X[:,1]