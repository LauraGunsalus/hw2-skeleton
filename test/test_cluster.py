from hw2skeleton import cluster
from hw2skeleton import io
import warnings
import os

def test_similarity():
    warnings.filterwarnings("ignore")
    pdbDir = os.getcwd() + "/data/"
    activeSites, activeSiteCoords = cluster.parsePDB(pdbDir)
    i, j = activeSites.index('276'),activeSites.index('4629')
    distanceMatrix = cluster.buildDistanceMatrix(activeSiteCoords)
    assert distanceMatrix[i][i] == 0 
    assert round(distanceMatrix[i][j],2) == 30.59

def test_partition_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]
    warnings.filterwarnings("ignore")
    pdbDir = os.getcwd() + "/data/"
    activeSitesI, activeSiteCoords = cluster.parsePDB(pdbDir)
    myIndices = list(map(lambda x: activeSitesI.index(str(x)), pdb_ids))
    active_sites = [activeSiteCoords[i] for i in myIndices]
    seed,k,maxIterations = 203,3,50
    kmeansAssignments = cluster.kmeans(active_sites, 2, seed, maxIterations)
    # update this assertion
    assert kmeansAssignments == {0: [2], 1: [0, 1]}

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]
    warnings.filterwarnings("ignore")
    pdbDir = os.getcwd() + "/data/"
    activeSitesI, activeSiteCoords = cluster.parsePDB(pdbDir)
    myIndices = list(map(lambda x: activeSitesI.index(str(x)), pdb_ids))
    active_sites = [activeSiteCoords[i] for i in myIndices]
    distanceMatrix = cluster.buildDistanceMatrix(active_sites)
    k = 2
    singleAssignments =  cluster.logToLabels(distanceMatrix, cluster.agglomerate(distanceMatrix, "single", k))
    completeAssignments = cluster.logToLabels(distanceMatrix, cluster.agglomerate(distanceMatrix, "complete", k))
    # update this assertion
    assert singleAssignments == {2: [2], 3: [0, 1]}
    assert completeAssignments == {2: [2], 3: [0, 1]}

