import numpy as np
from .base import BaseSelector

# -- Random Selector --
# Randomly select N samples from each class as prototypes
class RandomSelector(BaseSelector):
    
    def select_prototypes(self, X_train, y_train):
        np.random.seed(self.random_state)
        
        prototypes = []
        labels = []
        num_classes = len(np.unique(y_train))
        
        for class_id in range(num_classes):
            class_mask = (y_train == class_id)
            X_class = X_train[class_mask]
            
            num_select = min(self.n_cluster, len(X_class))
            indices = np.random.choice(len(X_class), num_select, replace=False)
            
            prototypes.append(X_class[indices])
            labels.extend([class_id] * num_select)
        
        return np.vstack(prototypes), np.array(labels)