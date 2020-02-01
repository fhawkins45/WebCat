from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

url = "https://www.npr.org/2020/01/29/800770355/pentagon-now-says-50-troops-not-34-suffered-brain-injuries-in-iran-strike"

#Read in data
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'talk.politics.guns']
news_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True)
news_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True)

#Give each word in the train data a unique number and training
count_vect = CountVectorizer()
X_train_tf = count_vect.fit_transform(news_train.data)
X_train_tf.shape


#Minimizing the weight of particular words (the, of, etc)
transformer = TfidfTransformer()
X_train_tfidf = transformer.fit_transform(X_train_tf)
X_train_tfidf.shape

#Using Naive Bayes Classifier
clf = MultinomialNB().fit(X_train_tfidf, news_train.target)
X_test_tf = count_vect.transform(news_test.data)
X_test_tfidf = transformer.transform(X_test_tf)
predicted = clf.predict(X_test_tfidf)

#web scraping
#page = requests.get(url)
#soup = BeautifulSoup(page.content, 'html.parser')
#text = soup.find('h1')


#docs_new = text
#X_new_counts = count_vect.transform(docs_new)
#X_new_tfidf = transformer.transform(X_new_counts)
#predicted = clf.predict(X_new_tfidf)


print("Accuracy:", accuracy_score(news_test.target, predicted))
print(metrics.classification_report(news_test.target, predicted, target_names=news_test.target_names))



