# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:29:21 2021

@author: 无冬之夜
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
def bayes():
    #1.获取数据
    date = fetch_20newsgroups(subset='all')
    #2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(date.data,date.target)
    #3.特征工程：特征抽取tfidf
    #实例一个估计器
    tfidf= TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train)
    #测试集只需标准化数据所以不需要再次拟合数据，因此直接使用transform实现标准化
    x_test = tfidf.transform(x_test)
    #4.baiyes预估器流程
    mlb = MultinomialNB(alpha=1.0)
    mlb.fit(x_train,y_train)
    #5.模型评估
    y_predict = mlb.predict(x_test)
    print(y_predict[:100])
    print(y_test[:100])
    print(mlb.score(x_test,y_test))
    
    return None
if __name__ == "__main__":
   bayes()