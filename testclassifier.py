import pandas as pd
import numpy as np
import string
import re
import random
from collections import Counter
from nltk.corpus import stopwords


pt1 = pd.read_csv('articles1.csv.zip',
                  compression='zip', index_col=0)
pt2 = pd.read_csv('articles2.csv.zip',
                  compression='zip', index_col=0)
pt3 = pd.read_csv('articles3.csv.zip',
                  compression='zip', index_col=0)

articles = pd.concat([pt1, pt2, pt3])


def clean_text(article):
    clean1 = re.sub(r'[' + string.punctuation + '’—”' + ']', "", article.lower())
    return re.sub(r'\W+', ' ', clean1)


articles['tokenized'] = articles['content'].map(lambda x: clean_text(x))
#print(articles['tokenized'].head())

articles['num_wds'] = articles['tokenized'].apply(lambda x: len(x.split()))
articles = articles[articles['num_wds']>0]

#print(articles['num_wds'].mean())
#print(articles['num_wds'].min())
random.shuffle(articles)

all_words = []

for a in articles():
	all_words.append(a.lower())

all_words = nltk.FreqDist(all_words)
word_featues = list(all_words.keys())[:3000]

def find_features(articles):
	words = set(articles)
	features = {}
	for w in word_featues:
		features[w] = (w in words)

	return features

print((find_features(articles.words('liberal/cv000_29416.txt'))))