import pandas as pd
import numpy as np
import optuna
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
# from preprocessing import preprocessor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import pickle
# from CustomWrapper import CustomHandler
from nan_handler import DropNaN
from preprocessing import process
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('completeSpamAssassin.csv')

# pro = process()

# dataset preprocessing

# nan_handle = nan_handler()


ct = ColumnTransformer(transformers=[
    ('nan', DropNaN(), df.columns)
])

dfpro = ct.fit_transform(df)

print(dfpro)