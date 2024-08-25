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
            # print(x)
            x = x.apply(self.process_function)
            return pd.DataFrame(x, index=[0])
except Exception as e:
    print(e)