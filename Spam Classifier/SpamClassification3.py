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
from preprocessing import process
from preprocessorWrapper import ProcessFunction
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

print(df.columns)

dfy = df['Label']
dfx = df.drop('Label', axis=1)

ct = ColumnTransformer(transformers=[
    ('drop_unnamed', 'drop', ['Unnamed: 0']),
    ('nan', DropNaN(), df.columns),
    ('preprocess', ProcessFunction(process)),
])

pipe = Pipeline([
    ('ct', ct),
    ('vectorizer', TfidfVectorizer()),
    ('model', RandomForestClassifier(n_estimators=50, random_state=0))
])



xtrain, xtest, ytrain, ytest = train_test_split(dfx, dfy, shuffle=True, test_size=0.25)
dfpro = pipe.fit(xtrain, ytrain)
predictions = pipe.predict(xtest)

# print(pd.DataFrame(dfpro).columns)
print(predictions)