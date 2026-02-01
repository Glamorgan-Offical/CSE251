import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from .base import BaseSelector


class ClusterSelector(BaseSelector):
    
    def __init__(self, num_prototypes_per_class=50, random_state=114514, pca_components=100):

        super().__init__(num_prototypes_per_class, random_state)
        self.pca_components = pca_components
        
    def select_prototypes(self, X_train, y_train):
        """
        使用PCA+K-means选择原型
        
        流程：
        1. 对每个类别分别处理
        2. PCA降维到低维空间
        3. K-means聚类
        4. 选择最接近聚类中心的原始样本
        
        Returns:
            prototypes: 选中的原型（原始784维空间）
            prototype_labels: 原型标签
        """
        prototypes = []
        prototype_labels = []
        num_classes = len(np.unique(y_train))
        
        for class_id in range(num_classes):
            # 获取该类的所有样本
            class_mask = (y_train == class_id)
            X_class = X_train[class_mask]
            
            # PCA降维
            n_components = min(self.pca_components, X_class.shape[0], X_class.shape[1])
            pca = PCA(n_components=n_components)
            X_class_reduced = pca.fit_transform(X_class)
            
            # K-means聚类
            num_clusters = min(self.num_prototypes_per_class, len(X_class))
            kmeans = KMeans(
                n_clusters=num_clusters,
                init='k-means++',
                random_state=self.random_state,
                n_init=10
            )
            kmeans.fit(X_class_reduced)
            
            # 找到最接近每个聚类中心的原始样本
            for center in kmeans.cluster_centers_:
                distances = np.linalg.norm(X_class_reduced - center, axis=1)
                closest_idx = np.argmin(distances)
                prototypes.append(X_class[closest_idx])  # 存储原始784维样本
                prototype_labels.append(class_id)
        
        return np.array(prototypes), np.array(prototype_labels)