from sklearn import metrics
from wals.wals_procedure import wals


class WALSestimator:

    def __init__(self, endog, focus, auxillary):
        self.endog = endog.to_numpy()
        self.focus = focus.to_numpy()
        self.auxillary = auxillary.to_numpy()

    def fit(self):
        self.b, self.se, self.V = wals(self.focus, self.auxillary, self.endog)

    def predict(self, x1, x2):
        self.predictions = x1@self.b[:x1.shape[1]] + x2@self.b[x1.shape[1]:]

    def mse(self, y_true):
        return metrics.mean_squared_error(y_true, self.predictions)
