#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import codecs
import argparse
import numpy as np
import pandas as pd
#from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import f1_score, accuracy_score

import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from matplotlib import pyplot

df = pd.read_csv("/Users/luyi/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/19f1b1ea31d72936bd394b1c12e52ac2/Message/MessageTemp/9a54283ea7df683eb8a76576ba128aaf/File/Website_Fact.csv")

df2 = pd.read_csv("/Users/luyi/Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/19f1b1ea31d72936bd394b1c12e52ac2/Message/MessageTemp/9a54283ea7df683eb8a76576ba128aaf/File/Website_Bias.csv")

df = df.iloc[:,2:]
df2 = df2.iloc[:,2:]
label_fact = df.fact.map({0:'low', 1:'mixed', 2:'high'})
label_bias3 = df2.bias.map({0:'right',1:'right',2:'center',3:'center',4:'center',5:'left',6:'left'})


def tsne_plot(c,f):
    data = df.filter(like=c)
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(data)
    labels = f
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        pyplot.scatter(x[i],y[i])
        pyplot.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    pyplot.show()


colums_name = ['body','title','wikicontent','wikisummary','wikicategories','wikitoc','description']
# %%time
for c in colums_name:
    figure(figsize=(20,10))
    tsne_plot(c,label_fact)

from yellowbrick.text import TSNEVisualizer
from sklearn.feature_extraction.text import TfidfVectorizer


def tsne_pack(c,l):
    my_title = "t-SNE Plot of "+c+" feature"
    data = df.filter(like=c)
    tfidf = TfidfVectorizer()
    new_values = tfidf.fit_transform(corpus)
    tsne = TSNEVisualizer(title=my_title)
    tsne.fit(data,l)
    tsne.poof()


label_fact = df.fact.map({0:'low', 1:'mixed', 2:'high'})
label_bias3 = df2.bias.map({0:'right',1:'right',2:'center',3:'center',4:'center',5:'left',6:'left'})

# %%time
for c in colums_name:
    figure(figsize=(20,10))
    plt.title(c)
    tsne_pack(c,label_fact)

# %%time
for c in colums_name:
    figure(figsize=(20,10))
    tsne_pack(c,label_bias3)

data = df.filter(like='title')
column_name = ['wikicontent','wikicategories','wikisummary','wikitoc','verified','has_wiki']
final = data.join(df.filter(like='wikicontent')).join(df.filter(like='wikicategories'))\
        .join(df.filter(like='wikisummary')).join(df.filter(like='wikitoc')).join(df.filter(like='has_twitter'))


def tsne(c,l):
    my_title = "t-SNE Plot of final model"
    data = c
    tfidf = TfidfVectorizer()
    new_values = tfidf.fit_transform(corpus)
    tsne = TSNEVisualizer(title=my_title)
    tsne.fit(data,l)
    tsne.poof()


# %%time
    figure(figsize=(20,10))
    tsne(final,label_bias3)

# %%time
    figure(figsize=(20,10))
    tsne(final,label_fact)


