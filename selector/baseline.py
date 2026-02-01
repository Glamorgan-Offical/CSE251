import numpy as np
from .base import BaseSelector


class FullSelector(BaseSelector):
    
    def __init__(self):
        super().__init__(num_prototypes_per_class=None, random_state=None)
    
    def select_prototypes(self, X_train, y_train):
        return X_train, y_train