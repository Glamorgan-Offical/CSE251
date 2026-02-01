from abc import ABC, abstractmethod
import time


class BaseSelector(ABC):
    
    def __init__(self, num_prototypes_per_class=50, random_state=114514):
        self.num_prototypes_per_class = num_prototypes_per_class
        self.random_state = random_state
        self.prototypes = None
        self.prototype_labels = None
        self.selection_time = None
        
    @abstractmethod
    def select_prototypes(self, X_train, y_train):
        """
        Abstract method to select a representative subset of prototypes from the training set.

        Args:
            X_train: The input training samples.
            y_train: The target labels for the training samples.

        Returns:
            prototypes: A subset of X_train chosen as prototypes.
            prototype_labels: The labels corresponding to the selected prototypes.
        """
        pass
    
    def fit(self, X_train, y_train):
        start_time = time.time()
        self.prototypes, self.prototype_labels = self.select_prototypes(X_train, y_train)
        self.selection_time = time.time() - start_time
        return self
    
    def get_info(self):
        return {
            'name': self.__class__.__name__,
            'num_prototypes': len(self.prototypes) if self.prototypes is not None else 0,
            'num_prototypes_per_class': self.num_prototypes_per_class,
            'selection_time': self.selection_time,
        }