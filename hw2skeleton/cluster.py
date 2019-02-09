import numpy as np
from Bio.PDB import PDBParser
import os
import random

##############################################################################
####  Objective 1: Implement a similarity metric to compare active sites.
##############################################################################

# Goal: Root-mean-square deviation of atomic positions
# Input: coordinates for backbone atoms of two proteins
# Output: rmsd between two proteins
def rmsd(atomA, atomB):
    backboneNo = len(atomA)
    total = 0
    for i in range(backboneNo):
        (x1, y1, z1) = atomA[i]
        (x2, y2, z2) = atomB[i]
        total += (x2-x1)**2 + (y2-y1)**2 + (z2 - z1)**2
    return np.sqrt(total/backboneNo)

# Goal: Get average coordinates for backbone atoms across all residues
# Input: biopython pdb structure for a protein
# Output: Average backbone coordinates for the protein
def getAverageCoordinates(structure):
    N_coords = []
    Ca_coords = []
    C_coords = []
    for residue in structure[0].get_residues():
        for atom in residue:
            if atom.get_name() == "N": N_coords.append(atom.get_coord())
            elif atom.get_name() == "CA": Ca_coords.append(atom.get_coord())
            elif atom.get_name() == "C": C_coords.append(atom.get_coord())
    return list(map(np.mean,list(zip(*N_coords)))),list(map(np.mean,list(zip(*Ca_coords)))),list(map(np.mean,list(zip(*C_coords))))

# Goal: Parse protein PDB files and extract average backbone coordinates for each protein
# Input: directory containing pdb files
# Output: PDB file names and average backbone atom locations for each protein
def parsePDB(pdbDir):
    pdbs = os.listdir(pdbDir)
    activeSites = []
    activeSiteCoords = []
    for pdb in pdbs:    
        label = os.path.splitext(pdb)[0]
        activeSites.append(label)
        parser = PDBParser()
        structure = parser.get_structure(label, pdbDir + pdb)
        activeSiteCoords.append(getAverageCoordinates(structure))
    return activeSites, activeSiteCoords

# Goal: compute similarity of all proteins
# Input: protein active site coords
# Output: distance matrix
def buildDistanceMatrix(activeSiteCoords):
    noProteins = len(activeSiteCoords)
    distanceM = np.zeros([noProteins, noProteins])
    for i in range(noProteins):
        for j in range(noProteins):
            distanceM[i][j] = rmsd(activeSiteCoords[i], activeSiteCoords[j])
    return distanceM

##############################################################################
####  Objective 2: Implement a partitioning algorithm.
##############################################################################

# K means across 3d space: backbone 
# Input: distance matrix, k clusters
# Output: final cluster assignments
def kmeans(activeSiteCoords, k, seed, maxIterations):
    random.seed(seed)
    # compute k random centers
    centers = [random.randint(0,len(activeSiteCoords)) for i in range(k)]
    centerCoords = [activeSiteCoords[i] for i in centers]
    newCenterCoords = None
    assignmentDict = assignClusters(activeSiteCoords, centerCoords, k)
    newCenterCoords = defineNewCentroids(assignmentDict, activeSiteCoords, centerCoords)
    iterations = 1
    while ((newCenterCoords != centerCoords) | (iterations < maxIterations)):
        centerCoords = newCenterCoords
        # assign points to closest cluster
        assignmentDict = assignClusters(activeSiteCoords, centerCoords, k)
        # find mean of each cluster
        newCenterCoords = defineNewCentroids(assignmentDict, activeSiteCoords, centerCoords)
        iterations += 1
    return assignmentDict

# Goal: assign all points to nearest cluster
# Input: list of all active site coordinates, list of centroid indices
# Output: list of closest centroid to each active site
def assignClusters(activeSiteCoords, centerCoords, k):
    assignmentDict = {l: [] for l in range(k)}
    for i in range(len(activeSiteCoords)):
        minDist, minI = float("Inf"), None
        for j in range(k):
            newDist = rmsd(activeSiteCoords[i], centerCoords[j])
            if (newDist < minDist):
                minDist, minI = newDist, j
        assignmentDict[minI].append(i)
    return assignmentDict

# Goal: find new centroids- mean of prevoius centroid
# Input: cluster membership, point coordinates
# Output: new centroid location
def defineNewCentroids(assignmentDict, activeSiteCoords, oldCenterCoords):
    newCoords = []
    for cluster, memberList in assignmentDict.items():
        if len(memberList) == 0: # no closest members
            newCoords.append(oldCenterCoords[cluster])
        else:
            averageMember = getAverageMember(memberList, activeSiteCoords)
            newCoords.append(averageMember)
    return newCoords

# Goal: Find mean cluster location
# Input: members of a cluster, cluster locations
# Output: average location of the cluster
def getAverageMember(memberList, activeSiteCoords):
    members = [activeSiteCoords[i] for i in memberList]
    n = len(members)
    memberDict = {'Nx':0, 'Ny':0, 'Nz':0, 
                 'Cax':0, 'Cay':0, 'Caz':0, 
                 'Cx':0, 'Cy':0, 'Cz':0}
    for member in members:
        ([x1,y1,z1],[x2,y2,z2],[x3,y3,z3]) = member
        memberDict['Nx'] += x1
        memberDict['Ny'] += y1
        memberDict['Nz'] += z1
        memberDict['Cax'] += x2
        memberDict['Cay'] += y2
        memberDict['Caz'] += z2
        memberDict['Cx'] += x3
        memberDict['Cy'] += y3
        memberDict['Cz'] += z3
    averageCoord = ([(memberDict['Nx'])/n, (memberDict['Ny'])/n, (memberDict['Nz'])/n],
            [(memberDict['Cax'])/n, (memberDict['Cay'])/n,(memberDict['Caz'])/n],
            [(memberDict['Cx'])/n, (memberDict['Cy'])/n,(memberDict['Cz'])/n])
    return averageCoord

# Goal: Get the average distance for each cluster
# Input: Dictionary containing list of distances for each cluster
# Output: Dictionary of mean distance for each centroid
def getCentroidMeans(distanceDict):
    averageDict = {}
    for cluster,dists in distanceDict.items():
        averageDict[cluster] = np.mean(dists)
    return averageDict

##############################################################################
####  Objective 3: Implement a hierarchical algorithm.
##############################################################################

# Goal: Cluster items via agglomerative clustering  
# Input: distance matrix, linkage metric, final number of clusters
# Output: list of merges of format (cluster 1, cluster 2, cluster distance, new cluster)
def agglomerate(distanceMatrix, metric, noClusters):
    clusterNames = np.arange(0, len(distanceMatrix))
    mergeHistory = []
    while(len(distanceMatrix)) > noClusters:
        distanceMatrix, clusterNames, mergeHistory = mergeTwoClusters(distanceMatrix, clusterNames, mergeHistory, metric)
    return mergeHistory

# Goal: transform log of merges to a list of cluster membership
# Input: distance matrix, list of merges
# Output: dictionary of cluster membership
def logToLabels(distanceMatrix, mergeHistory):
    clusterMembership = list(range(0, len(distanceMatrix)))
    for merge in mergeHistory:
        [clusterA, clusterB, distance, newCluster] = merge
        previousIs = [i for i in range(len(clusterMembership)) if ((clusterMembership[i] == clusterA) | (clusterMembership[i] == clusterB))]
        for i in previousIs:
            clusterMembership[i] = newCluster
    # Convert membership list to membership dictionary
    assignmentDict = {l: [] for l in set(clusterMembership)}
    for i in range(len(clusterMembership)):
        assignmentDict[clusterMembership[i]].append(i)
    return assignmentDict

# Goal: Merge two closest clusters 
# Input: distance matrix, list of cluster names, list of merges, linkage metric
# Output: merged distance matrix, new list of cluster names, list of merges
def mergeTwoClusters(distanceMatrix, clusterNames, mergeHistory, metric):
    [i, j], smallestDistance = getSmallestIndex(distanceMatrix)
    newCluster = max(clusterNames) + 1
    # Add merge to log
    mergeHistory.append([clusterNames[i], clusterNames[j], smallestDistance, newCluster]) 
        # cluster 1, cluster 2, merge distance, new cluster
        # Merged clusterNames[i] and clusterNames[j] with distance smallestDistance to make new cluster newCluster 
    # Update cluster name list 
    mask = np.ones(len(distanceMatrix),dtype=bool)
    mask[i], mask[j] = False,False
    result = clusterNames[mask]
    newClusterNames = np.append(result, newCluster)
    # Rebuild distance matrix
    newM = np.delete(distanceMatrix, (i,j),  axis = 0) 
    newM = np.delete(newM, (i,j), axis = 1) 
    newDistances = []
    for k in range(0, len(distanceMatrix)):
        if (k != i and k != j):
            if (metric == "single"):
                newDistances.append(min(distanceMatrix[k][i], distanceMatrix[k][j]))
            elif (metric == "complete"):
                newDistances.append(max(distanceMatrix[k][i], distanceMatrix[k][j]))
            else: 
                raise ValueError('Error: linkage method not recognized')
    newDistanceMatrix = np.column_stack([newM, newDistances])
    newDistances.append(0)
    newDistanceMatrix = np.row_stack([newDistanceMatrix, newDistances])
    return newDistanceMatrix, newClusterNames, mergeHistory

# Goal: Get index of minimum distance in distance matrix
    # Only check half of matrix since matrix is symmetrical 
# Input: distance matrix
# Output: Index of smallest distance between two clusters, smallest distance
def getSmallestIndex(distanceM):
    (smallestDistance, smallestI) = (float("Inf"), [float("Inf"),float("Inf")])
    for i in range(0, len(distanceM)): # rows
        for j in range(i + 1, len(distanceM)):    # columns
            distance = distanceM[i][j]
            if (distance < smallestDistance):
                (smallestDistance, smallestI) = (distance, [i,j])
    return smallestI, smallestDistance

##############################################################################
####  Objective 4: Implement a function to measure the quality of clusterings.
##############################################################################

# Goal: Find how similar each item is to other members of the same cluster
    # compared to items not in the same cluster
    # Ratio of 1 denotes that member is as close on average to all items than
    # to items in its own cluster. Larger ratios denote better clustering.
# Input: matrix of item similarity, dictionary of item cluster membership
# Output: Distance Ratio for each item
def compareDistance(distanceMatrix, membershipDict):
    distanceRatio = []
    for cluster, memberList in membershipDict.items():
        for member in memberList:
            clusterDist = getClusterDist(member, memberList,distanceMatrix)
            outgroupDist = getOutgroupDist(member, memberList, distanceMatrix)
            if outgroupDist == 0:
                distanceRatio.append(clusterDist)
            else:
                distanceRatio.append(clusterDist/outgroupDist)
    return distanceRatio

# Goal: Find average distance of item to items in the same cluster
# Input: current item, list of items in cluster, matrix of item similarity
# Output: average in-cluster distance
def getClusterDist(member, memberList, distanceMatrix):
    familyMembers = len(memberList) - 1 # number of other elements in cluster
    if familyMembers == 0:
        return 0 # there are no other elements in the cluster
    distanceSum = 0
    for familyMember in memberList:
        distanceSum += distanceMatrix[familyMember][member]
    return distanceSum/familyMembers
    # optimal: 0
    # worst case: Infinity

# Goal: Find average distance of item to items not in the same cluster
# Input: current item, list of items in cluster, matrix of item similarity
# Output: average out-cluster distance
def getOutgroupDist(member, memberList, distanceMatrix):
    outerTotal = 0 # number of outgroup elements
    outerSum = 0   # sum of outgroup element distances
    for i in range(len(distanceMatrix)):
        if (i not in memberList):
            outerSum += distanceMatrix[i][member]
            outerTotal += 1
    if outerTotal == 0: 
        return float("Inf")
    return outerSum/outerTotal
    # optimal # Infinity
    # worse case: 0

# Goal: compute cluster quality for both clusterings for all possible k
# Input: matrix of site similarity, site coordinates, random seed
# Output: average distance ratio for each possible cluster size for each algorithm
def compareClusters(distanceMatrix,activeSiteCoords,seed):
    kmeans_dist = []
    single_dist = []
    complete_dist = []
    for k in range(1, len(distanceMatrix)):
        kmeans_assignment = kmeans(activeSiteCoords, k, seed, 50)
        single_assignment = logToLabels(distanceMatrix, agglomerate(distanceMatrix, "single", k))
        complete_assignment = logToLabels(distanceMatrix, agglomerate(distanceMatrix, "complete", k))
        kmeans_dist.append(np.mean(compareDistance(distanceMatrix, kmeans_assignment))) 
        single_dist.append(np.mean(compareDistance(distanceMatrix, single_assignment)))
        complete_dist.append(np.mean(compareDistance(distanceMatrix, complete_assignment)))
        print("Completed for " + str(k) + " clusters")
    return kmeans_dist, single_dist, complete_dist


##############################################################################
####  Objective 5: Implement a function to compare your two clusterings
##############################################################################

# Goal: For two cluster assigments of k clusters, compute average group membership across clusters
# Input: cluster assignment with algorithm A, cluster assignment with algorithm B
# Output: Average membership of clusters from assignment A in clusters of assignment B 
def clusterMembership(assignmentA, assignmentB):
    totalList = []
    for clusterA in list(assignmentA.values()):
        if len(clusterA) > 0:
            propList = []
            for clusterB in list(assignmentB.values()):
                propList.append(sum([item in clusterB for item in clusterA])/len(clusterA))
            totalList.append(max(propList))
    return np.mean(totalList)

# Goal: compute cluster membership for both clusterings for all possible k
# Input: matrix of site similarity, site coordinates, random seed
# Output: average cluster membership for each possible cluster size for each algorithm
def compareclusterMembership(distanceMatrix,activeSiteCoords,seed):
    kmeans_single = []
    kmeans_complete = []
    single_complete = []
    for k in range(1, len(distanceMatrix)):
        kmeans_assignment = kmeans(activeSiteCoords, k, seed, 50)
        single_assignment = logToLabels(distanceMatrix, agglomerate(distanceMatrix, "single", k))
        complete_assignment = logToLabels(distanceMatrix, agglomerate(distanceMatrix, "complete", k))
        kmeans_single.append(clusterMembership(kmeans_assignment, single_assignment))
        kmeans_complete.append(clusterMembership(kmeans_assignment, complete_assignment))
        single_complete.append(clusterMembership(single_assignment, complete_assignment))
        print("Completed for " + str(k) + " clusters")
    return kmeans_single, kmeans_complete, single_complete






