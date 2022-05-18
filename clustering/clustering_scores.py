from __future__ import annotations

import numpy as np
import random
from scipy.sparse import issparse
from s_dbw import S_Dbw, SD
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score, calinski_harabasz_score, davies_bouldin_score

import time
from itertools import product
from math import log

# Scikit-Learn
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances

def divide(data, labels):
    clusters = set(labels)
    clusters_data = []
    for cluster in clusters:
        clusters_data.append(data[labels == cluster, :])
    return clusters_data

def get_centroids(clusters):
    centroids = []
    for cluster_data in clusters:
        centroids.append(cluster_data.mean(axis=0))
    return centroids

def cohesion(data, sse, **args):
    # ``` Basically mean sse ```
    return sse / data.shape[0]

def separation(data, sst, sse, **args):
    # calculate separation as mean SST - SSE
    return (sst - sse) / data.shape[0]

def SST(data, **args):
    c = get_centroids([data])
    return ((data - c) ** 2).sum()

def SSE(clusters, centroids, **args):
    result = 0
    for cluster, centroid in zip(clusters, centroids):
        result += ((cluster - centroid) ** 2).sum()
    return result

# Clear the store before running each time
within_cluster_dist_sum_store = {}
def within_cluster_dist_sum(cluster, centroid, cluster_id):
    if cluster_id in within_cluster_dist_sum_store:
        return within_cluster_dist_sum_store[cluster_id]
    else:
        result = (((cluster - centroid) ** 2).sum(axis=1)**.5).sum()
        within_cluster_dist_sum_store[cluster_id] = result
    return result

def RMSSTD(data, clusters, sse, **args):
    # sse=SSE(clusters, centroids)
    df = data.shape[0] - len(clusters)
    attribute_num = data.shape[1]
    return (sse / (attribute_num * df)) ** .5

# equal to separation / (cohesion + separation)
def RS(sst, sse, **args):
    return (sst - sse) / sst

def DB_find_max_j(clusters, centroids, i):
    max_val = 0
    max_j = 0
    for j in range(len(clusters)):
        if j == i:
            continue
        cluster_i_stat = within_cluster_dist_sum(clusters[i], centroids[i], i) / clusters[i].shape[0]
        cluster_j_stat = within_cluster_dist_sum(clusters[j], centroids[j], j) / clusters[j].shape[0]
        val = (cluster_i_stat + cluster_j_stat) / (((centroids[i] - centroids[j]) ** 2).sum() ** .5)
        if val > max_val:
            max_val = val
            max_j = j
    return max_val

def DB(data, clusters, centroids, **args):
    result = 0
    for i in range(len(clusters)):
        result += DB_find_max_j(clusters, centroids, i)
    return result / len(clusters)

def XB(data, centroids, sse, **args):
    # sse = SSE(clusters, centroids)
    min_dist = ((centroids[0] - centroids[1]) ** 2).sum()
    for centroid_i, centroid_j in list(product(centroids, centroids)):
        if (centroid_i - centroid_j).sum() == 0:
            continue
        dist = ((centroid_i - centroid_j) ** 2).sum()
        if dist < min_dist:
            min_dist = dist
    return sse / (data.shape[0] * min_dist)

def Sep(labels, k, dk, dist):
    clusters = sorted(set(labels))
    max_sep = None
    for cluster in clusters:
        cluster_data = dist[labels == cluster]
        cluster_data = cluster_data[:, labels != cluster]
        cluster_dk = dk[labels == cluster]
        sep = len(cluster_data[cluster_data <= np.c_[cluster_dk]]) / (k * cluster_data.shape[0])
        if max_sep is None or max_sep < sep:
            max_sep = sep
    return max_sep

def Com(labels, dist):
    clusters = sorted(set(labels))
    com = 0
    max_com = 0
    for cluster in clusters:
        cluster_data = dist[labels == cluster]
        cluster_data = cluster_data[:, labels == cluster]
        n_i = cluster_data.shape[0]
#        print(n_i, cluster_data.sum())
        if n_i > 1:
            cur_sum = 2 * cluster_data.sum() / (n_i * (n_i - 1))
            com += cur_sum
            if max_com < cur_sum:
                max_com = cur_sum
    return com, max_com

def CVNN(labels, k, dk, dist, **args):
    com, max_com = Com(labels, dist)
    return Sep(labels, k, dk, dist) + com / max_com

class ClusterData():
    def __init__(self, data=None, labels=None, k=None, dk=None, dist=None):
        self.data = data
        self.labels = labels
        self.k = k
        self.dk = dk
        self.dist = dist

        self.clusters = divide(self.data, self.labels)
        self.centroids = get_centroids(self.clusters)
        self.sst = SST(self.data)
        self.sse = SSE(self.clusters, self.centroids)
        self.scores = {
            'SST': self.sst, 'SSE': self.sse,
            'n_clusters': len(self.clusters),
            'noise_ratio': (self.labels == 0).sum() / len(self.labels)
        }

        self._metrics_functions = {
            'cohesion': cohesion,
            'separation': separation,
            'RMSSTD': RMSSTD,
            'RS': RS,
            'CH': calinski_harabasz_score,
            'silhouette_score': silhouette_score,
            'DB': davies_bouldin_score,
            'XB': XB,
            'SD': SD,
            'S_Dbw': S_Dbw,
            'CVNN': CVNN
        }

    def get_metrics_names(self):
        return [
            'cohesion', 'separation', 'RMSSTD', 'RS',
            'CH', 'silhouette_score', 'DB', 'XB',
            'SD', 'S_Dbw', 'CVNN'
        ]

    def get_scores(self, names, force=False):
        """
        names: list of strings with metric names to calculate
        """
        within_cluster_dist_sum_store.clear()

        for metricname in names:
            if (metricname in self.scores) and (not force):    # no recalculations
                continue
            if metricname in ['silhouette_score', 'SD', 'S_Dbw', 'CH', 'DB']:
                self.scores[metricname] = self._metrics_functions[metricname](self.data, self.labels)
            else:
                self.scores[metricname] = self._metrics_functions[metricname](**self.__dict__)
        return self
