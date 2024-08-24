from sklearn.base import BaseEstimator, TransformerMixin

class ProcessFunction:
    def __init__(self, process_function):
        self.process_function = process_function

    def fit(self, X, y=None):
        return self
    
    def transform(self, x):
        x['Body'] = x['Body'].apply(self.process_function)
        return x
    