# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:56:35 2021

@author: 大帅哥
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
def cf():
    #1.获取数据
    date = fetch_20newsgroups(subset='all')
    #2.数据集划分
    x_train,x_test,y_train,y_test = train_test_split(date.data,date.target)
    #3.特征工程
    #实例一个估计器
    tfidf= TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train)
    #之前已经用过fit实现了一个流程，后面不需要再用fit
    x_test = tfidf.transform(x_test)
    #4.模型训练
    mlb = MultinomialNB(alpha=1.0)
    mlb.fit(x_train,y_train)
    #5.模型评估
    y_predict = mlb.predict(x_test)
    print(y_predict[:100])
    print(y_test[:100])
    print(mlb.score(x_test,y_test))    
    return None
if __name__ == "__main__":
   cf()