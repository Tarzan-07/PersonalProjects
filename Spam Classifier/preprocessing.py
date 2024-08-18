import numpy as np
import pandas as pd 
import nltk
from nltk.stem.snowball import SnowballStemmer
# class preprocessor:
def process(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stemmer = SnowballStemmer('english')
    for i in text:
        y.append(stemmer.stem(i))
    
    return " ".join(y)