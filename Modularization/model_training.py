import numpy as np
from sklearn.base import clone

class ModelTrainer:
    def __init__(
        self,
        model:object,
        X:np.array,
        y:np.array,
        cv:object
        ):
        """ Initialises ModelTrainer class. """
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.best_model = None

    def train_model(
        self
        ):
        """ Trains model using cross-validation. """
        self.best_model = clone(self.model)  # Clone the model to make sure we're not altering the original model
        self.best_model.fit(self.X, self.y)
        return self.best_model