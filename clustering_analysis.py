from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

##############################################################################
####  Objective 1: Implement a similarity metric to compare active sites.
##############################################################################

# Example Calls:	
	# pdbDir = "/Users/lgunsalus/Desktop/Algorithms/hw2/active_sites/"
	# activeSites, activeSiteCoords = parsePDB(pdbDir)
	# distanceMatrix = buildDistanceMatrix(activeSiteCoords)

##############################################################################
####  Objective 2: Implement a partitioning algorithm.
##############################################################################

# Example Calls:
	# adict, acoords = kmeans(activeSiteCoords, 10, 203, 10)
	# kmeans(activeSiteCoords, 3, 203, 10)

##############################################################################
####  Objective 3: Implement a hierarchical algorithm.
##############################################################################


def computerHierarchyPlot(mergeHistory)
	hierarchy.dendrogram(mergeHistory)
	hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
	fig, axes = plt.subplots(1, 2, figsize=(8, 3))
	dn1 = hierarchy.dendrogram(mergeHistory, ax=axes[0], above_threshold_color='y',orientation='top')
	dn2 = hierarchy.dendrogram(mergeHistory, ax=axes[1],above_threshold_color='#bcbddc',orientation='right')
	hierarchy.set_link_color_palette(None)  # reset to default after use
	plt.show()

# Example Calls:
	# mergeHistory = agglomerate(dist, 'single', 1)
	# computerHierarchyPlot(mergeHistory)

##############################################################################
####  Objective 4: Implement a function to measure the quality of clusterings.
##############################################################################

def compareClusterQualityPlot(distanceMatrix,activeSiteCoords, seed):
	kmeans_dist, single_dist, complete_dist = compareClusters(distanceMatrix,activeSiteCoords,seed)
	k = list(range(1, len(distanceMatrix)))
	plt.plot(k, kmeans_dist)
	plt.plot(k, single_dist)
	plt.plot(k, complete_dist)
	plt.xlabel("Cluster Size (k)")
	plt.ylabel("Average Cluster Quality Score")
	plt.legend(loc='best')
	plt.legend(['K-means', 'Hierarchical (Single Linkage)', 'Hierarchical (Complete Linkage)'], loc='best')
	plt.title("Comparison of Clustering Methods Across All K")
	plt.show()

##############################################################################
####  Objective 5: Implement a function to compare your two clusterings
##############################################################################

def compareClusterMembershipPlot(distanceMatrix,activeSiteCoords, seed):
	kmeans_single, kmeans_complete, single_complete = compareClusterMembership(distanceMatrix,activeSiteCoords,seed) 
	k = list(range(1, len(distanceMatrix)))
	plt.plot(k, kmeans_single)
	plt.plot(k, kmeans_complete)
	plt.plot(k, single_complete)
	plt.xlabel("Cluster Size (k)")
	plt.ylabel("Average Cluster Membership Score")
	plt.legend(loc='best')
	plt.legend(['K-means/Hierarchical (Single Linkage)', 'K-means/Hierarchical (Complete Linkage)', 'Hierarchical (Single Linkage)/Hierarchical (Complete Linkage)'], loc='best')
	plt.title("Comparison of Clustering Membership Across All K")
	plt.show()


