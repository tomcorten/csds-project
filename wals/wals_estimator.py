from sklearn import metrics
from wals.wals_procedure import wals


class WALSestimator:

    def __init__(self, endog, focus, auxillary):
        self.endog = endog.to_numpy()
        self.focus = focus.to_numpy()
        self.auxillary = auxillary.to_numpy()

    def fit(self):
        self.b, self.se, self.V = wals(self.focus, self.auxillary, self.endog)

    def predict(self, X):
        self.predictions = X@self.b

    def mse(self, y_true):
        return metrics.mean_squared_error(y_true, self.predictions)
