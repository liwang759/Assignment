# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:29:00 2021

@author: PJC
"""

# 1.获取数据 from sklearn.datasets import fetch_20newsgroups
# 2.划分数据集
# 3特征工程:特征抽取
# 4.贝叶斯预估器流程 from sklearn.naive_bayes import MultinomialNB
# 5.模型评估

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def baiyes():
    # 1.获取数据
    data = fetch_20newsgroups(subset ='all')
    # 2.划分数据集
    x_train,x_test,y_train,y_test = train_test_split(data.data,data.target)
    # 3特征工程:特征抽取
     # 实例化一个估计器
    tf = TfidfVectorizer()
    x_train =  tf.fit_transform(x_train)
     # 测试集用transform实现标准化
    x_test = tf.transform(x_test)
    # 4.贝叶斯预估器流程
    mlb = MultinomialNB(alpha = 1.0)
    mlb.fit(x_train,y_train)
    # 5.模型评估
    y_predict = mlb.predict(x_test)
    print(y_predict[:100])
    print(y_test[:100])
    print(mlb.score(x_test,y_test))
    
    
    return None

#执行函数
if __name__ == "__main__":
   baiyes()
    