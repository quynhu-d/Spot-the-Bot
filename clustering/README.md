Wishart.py --- Wishart clustering
WishartFUZZY.py --- Wishart clustering on fuzzy number (with precomputed distance matrix)
clustering_scores.py --- Internal validation clustering metrics

Classifier results:
- kmeans_clf --- K-Means
- fcmeans_clf --- Fuzzy C-Means
- wishart_clf --- Wishart
- fws_clf --- Wishart on fuzzy data

Pipeline: clustering -> calculation of average/max/min intracluster distances -> classification by distances with SVM
