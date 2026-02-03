import numpy as np
from .base import BaseSelector

# -- Baseline Selector --
# Using all training set data as prototypes
class BaselineSelector(BaseSelector):
    
    def __init__(self):
        super().__init__(n_cluster=None, random_state=None)
    
    def select_prototypes(self, X_train, y_train):
        return X_train, y_train