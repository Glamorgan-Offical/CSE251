import numpy as np
from .base import BaseSelector


class RandomSelector(BaseSelector):
    
    def select_prototypes(self, X_train, y_train):
        """
        从每个类别随机选择样本
        
        Returns:
            prototypes: 随机选中的原型
            prototype_labels: 原型标签
        """
        np.random.seed(self.random_state)
        
        prototypes = []
        prototype_labels = []
        num_classes = len(np.unique(y_train))
        
        for class_id in range(num_classes):
            # 获取该类的所有样本
            class_mask = (y_train == class_id)
            X_class = X_train[class_mask]
            
            # 随机选择
            num_select = min(self.num_prototypes_per_class, len(X_class))
            indices = np.random.choice(len(X_class), num_select, replace=False)
            
            prototypes.append(X_class[indices])
            prototype_labels.extend([class_id] * num_select)
        
        return np.vstack(prototypes), np.array(prototype_labels)