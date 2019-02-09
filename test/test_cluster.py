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
    kmeansAssignments = cluster.kmeans(activeSiteCoords, k, seed, maxIterations)
    # update this assertion
    assert kmeansAssignments == {0: [0, 2, 3, 4, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 23, 24, 25, 28, 29, 31, 32, 33, 36, 38, 40, 42, 45, 46, 47, 51, 54, 55, 56, 57, 58, 60, 62, 63, 65, 66, 69, 70, 72, 75, 76, 77, 79, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91, 92, 94, 97, 98, 99, 100, 101, 102, 104, 107, 108, 109, 112, 113, 114, 115, 116, 117, 119, 120, 121, 123, 124, 127, 130, 131, 132, 133], 1: [15, 26, 95, 106, 110, 111, 135], 2: [1, 5, 7, 14, 20, 21, 22, 27, 30, 34, 35, 37, 39, 41, 43, 44, 48, 49, 50, 52, 53, 59, 61, 64, 67, 68, 71, 73, 74, 78, 83, 89, 93, 96, 103, 105, 118, 122, 125, 126, 128, 129, 134]}

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]
    warnings.filterwarnings("ignore")
    pdbDir = os.getcwd() + "/data/"
    activeSitesI, activeSiteCoords = cluster.parsePDB(pdbDir)
    myIndices = list(map(lambda x: activeSitesI.index(str(x)), pdb_ids))
    active_sites = [activeSiteCoords[i] for i in myIndices]
    distanceMatrix = cluster.buildDistanceMatrix(activeSiteCoords)
    k = 3
    singleAssignments =  cluster.logToLabels(distanceMatrix, cluster.agglomerate(distanceMatrix, "single", k))
    completeAssignments = cluster.logToLabels(distanceMatrix, cluster.agglomerate(distanceMatrix, "complete", k))
    # update this assertion
    assert singleAssignments == {265: [15, 26, 95, 106, 111, 135], 268: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134], 70: [70]}
    assert completeAssignments == {70: [70], 268: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134], 262: [15, 26, 95, 106, 110, 111, 135]}

