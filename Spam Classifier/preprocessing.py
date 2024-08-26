import numpy as np
import pandas as pd 
import nltk
from nltk.stem.snowball import SnowballStemmer
from loguru import logger
# class preprocessor:
stemmer = SnowballStemmer('english')
try:
    def process(text):
        df = []
        print(text)
        for i in text:
            if isinstance(i, int):
                logger.error("not a string, encountered int.")
                break
            i = str(i).lower()
            # print(type(i))
            l = nltk.word_tokenize(i)
            # print(type(l))
            y = []
            for i in l:
                if i.isalnum():
                    y.append(stemmer.stem(i))

            df.append(" ".join(y[:]))
            # y.clear()

            
        # for t in i:
        #     y.append(stemmer.stem(i))
        
        # print(df)
        
        # for i in df:
        # # print(df)
        #     dfnew.append(" ".join(i))
        
        print("preprocessing done")
        print(len(df))
        return df
    
    def spaces(df):
        # print(df)
        df = df.replace(r'^\s*$', np.nan, regex=True).dropna()
        print(df.isna().sum())
        return df

    def process2(text):
        df = []
        for i in text:
            if not isinstance(i, str):
                logger.warning(f"Non-string value encountered: {i}. Converting to string.")
                i = str(i)  # Convert non-string values to string
            
            i = i.lower()
            l = nltk.word_tokenize(i)
            y = [stemmer.stem(word) for word in l if word.isalnum()]
            
            df.append(" ".join(y))
        
        print("Preprocessing done")
        print(f"Processed {len(df)} records.")
        return df

    def ensureString(text):
        print("ensuring only string exists......")
        for i in text:
            if isinstance(i, int):
                print("int is present at {}".format(i))
                return False

        print("only strings are present")    
        return text
    
except Exception as e:
    print(e)
