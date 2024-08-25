import numpy as np
import pandas as pd 
import nltk
from nltk.stem.snowball import SnowballStemmer
# class preprocessor:
stemmer = SnowballStemmer('english')
try:
    def process(text):
        df = []
        print((text))
        for i in text:
            i = i.lower()
            # print(type(i))
            l = nltk.word_tokenize(i)
            # print(type(l))
            y = []
            for i in l:
                if i.isalnum():
                    y.append(stemmer.stem(i))

            df.append(" ".join(y[:]))
            y.clear()

            
        # for t in i:
        #     y.append(stemmer.stem(i))
        
        # print(df)
        
        # for i in df:
        # # print(df)
        #     dfnew.append(" ".join(i))
        
        print("preprocessing done")
        
        return df
    
    def spaces(df):
        # print(df)
        df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
        # print(df)
        return df
except Exception as e:
    print(e)