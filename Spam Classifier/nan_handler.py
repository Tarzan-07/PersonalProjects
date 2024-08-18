import numpy as np
import pandas as pd

def nan_handler(x):
    if isinstance(x, pd.DataFrame):
        arr = np.array(x.isnull().sum())
        if sum(arr) != 0:
            x.dropna(inplace=True, axis=0)
    else:
        raise TypeError