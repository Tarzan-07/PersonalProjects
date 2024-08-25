import numpy as np
import pandas as pd 
import nltk
from nltk.stem.snowball import SnowballStemmer
# class preprocessor:
stemmer = SnowballStemmer('english')
try:
    def process(text):
        print(type(text))
        for i in text:
            i = i.lower()
            # print(type(i))
            l = nltk.word_tokenize(i)
            # print(type(l))
            y = []
            for i in l:
                if i.isalnum():
                    y.append(stemmer.stem(i))

            i = y[:]
            y.clear()

            
        # for t in i:
        #     y.append(stemmer.stem(i))
        
        print("preprocessing done")
        return " ".join(y)
except Exception as e:
    print(e)