from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
try:
    class ProcessFunction:
        def __init__(self, process_function):
            self.process_function = process_function

        def fit(self, X, y=None):
            return self
        
        def transform(self, x):
            x = pd.DataFrame(x)
            x = x.apply(self.process_function)
            # print(x)
            return pd.DataFrame(x)
except Exception as e:
    print(e)