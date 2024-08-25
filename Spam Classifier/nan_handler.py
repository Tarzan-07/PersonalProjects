import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

try:
    class DropNaN(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            X = pd.DataFrame(X)
            if isinstance(X, pd.DataFrame):
                return X.dropna(axis=0)
            else:
                raise ValueError("Input should be a pandas dataframe")
except Exception as e:
    print(e)