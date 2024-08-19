from sklearn.base import BaseEstimator, TransformerMixin

class CustomHandler(BaseEstimator, TransformerMixin):
    def __init__(self, func) -> None:
        self.func = func

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        xtransformed = self.func(x)
        return xtransformed
    