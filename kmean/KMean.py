import numpy as np


class KMean:
    def __init__(self, K, max_iter=100):
        self.K = K
        self.max_iter = max_iter
        self.centroids = []

    def predict(self, X):
        n_sample, n_feature = X.shape
        # Initialize centroids
        self.centroids = X[np.random.choice(n_sample, self.K, replace=False)]

        for _ in range(self.max_iter):
            clusters = self._form_clusters(X)

            old_centroids = self.centroids
            self.centroids = self._update_centroid(clusters, X)

            if np.allclose(old_centroids, self.centroids):
                break
        return self._get_cluster_labels(X)

    def _form_clusters(self,X):
        clusters = [[] for _ in range(self.K)]
        for idx, x in enumerate(X):
            centroid_idx = self._find_closest_centroid(x)
            clusters[centroid_idx].append(idx)
        return clusters

    def _find_closest_centroid(self, x):
        return np.argmin([self.euclidean_dist(c, x) for c in self.centroids])

    def _update_centroid(self, clusters, X):
        return [np.mean(X[cluster], axis=0) if cluster else np.full(X.shape[1], None) for cluster in clusters]

    @staticmethod
    def euclidean_dist(x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))


    def _get_cluster_labels(self, X):
        return [self._find_closest_centroid(x) for x in X]
