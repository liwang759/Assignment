# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:24:54 2021

@author: asus
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
def bayes():
    #1.获取新闻的20个类别的数据
    date = fetch_20newsgroups(subset='all')
    #2.对数据集进行划分
    x_train,x_test,y_train,y_test = train_test_split(date.data,date.target)
    #3.特征工程：特征抽取tfidf
    #实例一个估计器
    tfidf= TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train)
    #测试集不需要再次拟合数据，因此可以直接使用transform实现标准化
    x_test = tfidf.transform(x_test)
    #4.baiyes预估器流程
    mlb = MultinomialNB(alpha=1.0)
    mlb.fit(x_train,y_train)
    #5.进行预测
    y_predict = mlb.predict(x_test)
    print(y_predict[:100])
    print(y_test[:100])
    print(mlb.score(x_test,y_test))
    
    return None
if __name__ == "__main__":
   bayes()