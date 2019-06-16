##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################

from util.VisualizeDataset import VisualizeDataset
from Chapter5.DistanceMetrics import InstanceDistanceMetrics
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
from Chapter5.Clustering import NonHierarchicalClustering
from Chapter5.Clustering import HierarchicalClustering
import copy
import pandas as pd
import matplotlib.pyplot as plot
import util.util as util


# Of course we repeat some stuff from Chapter 3, namely to load the dataset

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = './intermediate_datafiles/'

try:
    dataset = pd.read_csv(dataset_path + 'chapter4_result.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e
dataset.index = dataset.index.to_datetime()

# First let us use non hierarchical clustering.

clusteringNH = NonHierarchicalClustering()

# Let us look at k-means first.
#
k_values = range(2, 10)
silhouette_values = []
#
## Do some initial runs to determine the right number for k
#
print '===== kmeans clustering ====='
for k in k_values:
    print 'k = ', k
    dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'default', 20, 10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print 'silhouette = ', silhouette_score
    silhouette_values.append(silhouette_score)

plot.plot(k_values, silhouette_values, 'b-')
plot.xlabel('k')
plot.ylabel('silhouette score')
plot.ylim([0,1])
plot.show()

# And run the knn with the highest silhouette score

print '\n The highest average silhouette score is  '  + str(max(silhouette_values))
best_k = silhouette_values.index(max(silhouette_values))+2
print 'which you get from k =   '   +  str(best_k)

k = best_k

dataset_knn = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'default', 50, 50)
DataViz.plot_clusters_3d(dataset_knn, ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], 'cluster', ['label'])
DataViz.plot_silhouette(dataset_knn, 'cluster', 'silhouette')
util.print_latex_statistics_clusters(dataset_knn, 'cluster', ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], 'label')
del dataset_knn['silhouette']

#
k_values = range(2, 10)
silhouette_values = [[],[], [] ,[],[], [] ,[],[]]

# Do some initial runs to determine the right number for k

nr_of_iterations = 15
print '===== k medoids clustering ====='
for iteration in range(nr_of_iterations):
    print ' ------------------This is iteration number:    ' + str(iteration)
    for k in k_values:
        print 'k = ', k
        dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'default', 20, n_inits=10)
        silhouette_score = dataset_cluster['silhouette'].mean()
        print 'silhouette = ', silhouette_score
        silhouette_values[k-2].append(silhouette_score)

for i in range(len(silhouette_values)):
    silhouette_values[i] = sum(silhouette_values[i]) / len(silhouette_values[i])

plot.plot(k_values, silhouette_values, 'b-')
plot.ylim([0,1])
plot.xlabel('k')
plot.ylabel('silhouette score')
plot.show()


print '\n The highest average silhouette score is  '  + str(max(silhouette_values))
best_k = silhouette_values.index(max(silhouette_values))+2
print 'which you get from k =   '   +  str(best_k)
k = best_k

# And run k medoids with the highest silhouette score



dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'default', 20, n_inits=50)
DataViz.plot_clusters_3d(dataset_kmed, ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], 'cluster', ['label'])
DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette')
util.print_latex_statistics_clusters(dataset_kmed, 'cluster', ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], 'label')

# And the hierarchical clustering is the last one we try
#
clusteringH = HierarchicalClustering()
#
k_values = range(2, 10)
silhouette_values = []
#
# # Do some initial runs to determine the right number for the maximum number of clusters.
#
# exit()

print '===== agglomaritive clustering ====='
for k in k_values:
    print 'k = ', k
    dataset_cluster, l = clusteringH.agglomerative_over_instances(copy.deepcopy(dataset), ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z'], k, 'euclidean', use_prev_linkage=True, link_function='ward')
    silhouette_score = dataset_cluster['silhouette'].mean()
    print 'silhouette = ', silhouette_score
    silhouette_values.append(silhouette_score)
    if k == k_values[0]:
        DataViz.plot_dendrogram(dataset_cluster, l)

plot.plot(k_values, silhouette_values, 'b-')
plot.ylim([0,1])
plot.xlabel('max number of clusters')
plot.ylabel('silhouette score')
plot.show()

print '\n The highest average silhouette score is  '  + str(max(silhouette_values))
best_k = silhouette_values.index(max(silhouette_values))+2
print 'which you get from k (=max_clusters) set to:    '   +  str(best_k)


# And we select the outcome dataset of the knn clustering....

print '\n'
print 'Finished! Note though that the dendogram is based on k=2. You might wanna manually run agglomaritive clustering with the best k found.'

# dataset_knn.to_csv(dataset_path + 'chapter5_result.csv')
