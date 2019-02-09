import sys
import warnings
import os
from .io import read_active_sites, write_clustering, write_mult_clusterings
#from .cluster import cluster_by_partitioning, cluster_hierarchically
from hw2skeleton import cluster

# Some quick stuff to make sure the program is called correctly
if len(sys.argv) < 4:
    print("Usage: python -m hw2skeleton [-P| -H] <pdb directory> <output file>")
    sys.exit(0)

warnings.filterwarnings("ignore")
activeSites, activeSiteCoords = cluster.parsePDB(sys.argv[2])
distanceMatrix = cluster.buildDistanceMatrix(activeSiteCoords)

# Choose clustering algorithm
if sys.argv[1][0:2] == '-P':
    print("Clustering using Partitioning method")
    seed,k,maxIterations = 203,3,50
    kmeansAssignments = cluster.kmeans(activeSiteCoords, k, seed, maxIterations)
    write_clustering(sys.argv[3], list(kmeansAssignments.values()))

if sys.argv[1][0:2] == '-H':
    print("Clustering using hierarchical method")
    k = 3
    singleAssignments =  cluster.logToLabels(distanceMatrix, cluster.agglomerate(distanceMatrix, "single", k))
    write_clustering(sys.argv[3], list(singleAssignments.values()))
