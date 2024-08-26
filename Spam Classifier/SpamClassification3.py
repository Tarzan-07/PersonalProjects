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
from preprocessing import process, spaces, process2, ensureString
from preprocessorWrapper import ProcessFunction
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import pickle
# from CustomWrapper import CustomHandler
from nan_handler import DropNaN
from CustomWrapper import DropColumns
from preprocessing import process
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('completeSpamAssassin.csv')

print(df.columns)

dfy = df['Label']
dfx = df.drop('Label', axis=1)
ct = ColumnTransformer(transformers=[
    ('drop_unnamed', DropColumns(columns=['Unnamed: 0']), slice(0, None)),
])

# dfprocessed = ct.fit_transform(df)

pipe = Pipeline([
    ('ct', ct),
    ('drop', DropNaN()),
    ('prep', ProcessFunction(spaces)),
    ('ensure', ProcessFunction(ensureString)),
    ('preprocess', ProcessFunction(process2)),
    # ('vectorizer', TfidfVectorizer(token_pattern='(?u)\\b\\w+\\b')),
    # ('model', RandomForestClassifier(n_estimators=50, random_state=0))
])

# print(dfprocessed)

xtrain, xtest, ytrain, ytest = train_test_split(dfx, dfy, shuffle=True, test_size=0.25)
dfpro = pipe.fit(pd.DataFrame(xtrain), pd.DataFrame(ytrain))
# predictions = pipe.predict(xtest)

# print(pd.DataFrame(dfpro).columns)
# print(predictions)