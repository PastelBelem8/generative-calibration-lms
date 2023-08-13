from scipy.optimize import curve_fit

import numpy as np


def logistic(x, w, b):
    y = 1 / (1 + np.exp(-(w * x + b)))
    return y


class LogisticRegression:
    def __init__(self, init=None):
        self.parameters = init
        self._cov = None

    def fit(self, x, y, **kwargs):
        kwargs = {}
        if self.parameters:
            kwargs.update(p0=self.parameters)
        self.parameters, self._cov = curve_fit(logistic, x, y, **kwargs)

    def predict(self, x, threshold=0.5):
        return (logistic(x, *self.parameters) > threshold).astype(int)

    def predict_proba(self, x):
        return logistic(x, *self.parameters)
