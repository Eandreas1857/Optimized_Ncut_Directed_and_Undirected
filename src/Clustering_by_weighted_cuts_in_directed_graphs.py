import networkx as nx
import scipy.linalg
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
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
        nodelist = list(G.nodes())

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
        nodelist = list(G.nodes())

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
    L = (1/2)*np.dot(np.dot(T1,H),T1)
    return L

def k_smallest_eigvec(nodelist, L, k):
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
        Matrix Y, where Y has k smallest eigenvectors of nL as a column vectors.

    Notes:
    ------
    The columns of Y are automatically orthogonal, since nL is real-valued, 
    symmetric matric. Using the Variant to the BestWCut algorthm, so Y is 
    normalized to have rows of length 1. Using L2 norm.  

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
 
    return preprocessing.normalize(Y, norm="l2")

def partition_G(nodelist, Y):
    """Returns a partition of the graph G basied on k smallest eigenvectors
    of the Hermitian part of normalized Laplacian matrix nL.

    Parameters
    ----------
    nodelist : collection
        A collection of nodes in `G`. 
    
    Y : array 
        k smallest eigenvectors of nL as columns of Y, see k_smallest_eigvec.

    Returns
    -------
    clusters : dict
        Cluster 0...k-1 as keys, as set of nodes in G in cluster i = 0,..,k-1 as values.
    
    Notes:
    ------
    Using kmeans algorthm to cluster groups. See [1] for more information.

    References
    ----------
    .. [1] Marina Meila and Jianbo Shi.
           *A Random Walks View of Spectral Segmentation*.
           <https://sites.cs.ucsb.edu/~veronika/MAE/arandomwalksviewofimgsegmt_meila_shi_nips00.pdf>

    """
    k = len(Y[0])
    kmeans = KMeans(n_clusters=k)
    pred_y = kmeans.fit_predict(Y)

    clusters = {}
    for i in range(len(pred_y)):
        if pred_y[i] not in clusters:
            clusters[pred_y[i]] = [nodelist[i]]
        else:
            clusters[pred_y[i]] += [nodelist[i]]

    return clusters

def indicator_vector(nodelist, cluster_list):
    nlen = len(nodelist)
    mlen = len(cluster_list)

    X = np.full((nlen, mlen), np.nan, order=None)

    for i in range(len(cluster_list)):
        Ci = [1  if n in cluster_list[i] else 0 for n in nodelist ]
        X[:,i] = Ci

    X = np.asarray(X, dtype=None)
    return X

def WCut(D, W, T, X):
    """Returns weighted cut of graph partition into clusters C = {C1,...,Ck} [1].

    Parameters
    ----------
    D : array
        Out-Degree diagonal matrix, see asym_weighted_degree_matrix.

    W : array
        Weight matrix, see asym_weighted_degree_matrix.
    
    T : array
        User defined weighting of the nodes. For WNcut introduced in [1], use T = D.

    X : array
        The indicator vector of a cluster C = {C1,...,Ck} where the ith column of X represents
        the cluster Ci. Expects X[i][j] if jth node is in cluster Ci and 0 otherwise.

    Returns
    -------
    number
        Returns weighted cut WCut(C) of cluster C = {C1,...,Ck} as introduced in [1].

    References
    ----------
    .. [1] Marina Meila and William Pentney.
           *Clustering by weighted cuts in directed graphs*.
           <https://sites.stat.washington.edu/mmp/Papers/sdm-wcuts.pdf>

    """   
    A = D-W
    
    cut = 0
    for i in range(len(X[0])):

        top = np.dot(X[:,i].transpose(), np.dot(A, X[:,i]))
        bottom = np.dot(X[:,i].transpose(), np.dot(T, X[:,i]))

        cut += top/bottom
    
    return cut

def asym_optimized_normalized_cut(G, k, nodelist = None, data = None, return_clusters = True):
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

    k : int
        Number of desired clusters.
    
    Returns
    -------
    Y : array
        k smallest eigenvectors of nL.
    
    cut : number
        Optimied weighted cut (WCut) of clusters.

    clusters : dict
        Cluster 0...k-1 as keys, as set of nodes in G in cluster i = 0,..,k-1 as values.
        Only returned is return_clusters = True

    """
    if nodelist == None:
        nodelist = list(G.nodes())
    W = asym_weight_matrix(G, nodelist, data)
    D = asym_weighted_degree_matrix(G, nodelist, data)
    L = Hermitian_normalized_Laplacian(D, W, D)
    Y = k_smallest_eigvec(nodelist, L, k)

    clusters = partition_G(nodelist, Y)
    cluster_list = list(clusters.values())
    V = indicator_vector(nodelist, cluster_list)
    cut = WCut(D, W, D, V)
    return (Y, cut, clusters) if return_clusters == True else (Y, cut)
