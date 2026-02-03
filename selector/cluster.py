import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .base import BaseSelector

# -- Cluster Selector --
# PCA dimensionality reduction + K-means clustering for prototype selection
class ClusterSelector(BaseSelector):
    
    def __init__(self, n_cluster=50, random_state=114514, pca_components=100):

        super().__init__(n_cluster, random_state)
        self.pca_components = pca_components

    def select_prototypes(self, X_train, y_train):

        prototypes = []
        labels = []
        num_classes = len(np.unique(y_train))
        
        for class_id in range(num_classes):
            class_mask = (y_train == class_id)
            X_class = X_train[class_mask]
            
            # PCA
            n_components = min(
                self.pca_components, 
                X_class.shape[0], 
                X_class.shape[1]
            )
            pca = PCA(
                n_components=n_components, 
                random_state=self.random_state
            )
            X_class_reduced = pca.fit_transform(X_class)
            
            # K-means clustering
            num_clusters = min(self.n_cluster, len(X_class))
            kmeans = KMeans(
                n_clusters=num_clusters,
                init='k-means++',
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(X_class_reduced)
            
            # Find the sample closest to cluster center
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(X_class_reduced - center, axis=1)
                prototypes.append(X_class[np.argmin(distances)])
                labels.append(class_id)
        
        return np.array(prototypes), np.array(labels)