from sklearn.base import clone

class ModelTrainer:
    def __init__(self, model, X, y, cv):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv
        self.best_model = None

    def train_model(self):
        self.best_model = clone(self.model)  # Clone the model to make sure we're not altering the original model
        self.best_model.fit(self.X, self.y)
        return self.best_model