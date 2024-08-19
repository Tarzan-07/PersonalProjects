import numpy as np
import pandas as pd

def nan_handler(x):
    x.dropna(inplace=True, axis=0)
    return x