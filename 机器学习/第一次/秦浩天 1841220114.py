# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 18:21:07 2021

@author: 无冬之夜
"""

import pandas as pd
from sklearn.decomposition import PCA

def pca_demo():
    
    #1.获取数据
    products = pd.read_csv("D:/学习/机器学原理数据/数据/instacart/products.csv")
    orders = pd.read_csv("D:/学习/机器学原理数据/数据/instacart/orders.csv")
    order_products__prior = pd.read_csv("D:/学习/机器学原理数据/数据/instacart/order_products__prior.csv")
    aisles = pd.read_csv("D:/学习/机器学原理数据/数据/instacart/aisles.csv")

    #2.合并表并将其分组从而得到所需数据user_id与aisle
    #orders表数据中order_id项与order_products_prior表中的数据相同，所以可以通过order_id数据将两表连接合并为tab1
    tab1 = pd.merge(orders,order_products__prior,on=["order_id","order_id"])
    #products表中的product_id与tab1中的order_products__prior的product_id数据相同，即可通过product_id数据将两表连接合并为tab2
    tab2 = pd.merge(tab1,products,on=["product_id","product_id"])
    #同tab2中的products与tab1相连接原理，将tab2中的products数据与aisles数据通过aisle_id相连得出tab3
    tab3 = pd.merge(tab2,aisles,on=["aisle_id","aisle_id"])
    #将结合所有数据后的表tab3中的user_id与aisle进行分组
    tab4 = pd.crosstab(tab3["user_id"], tab3["aisle"])

    #3.利用主成分分析方法进行降为
    #实例化一个PCA转化器,将结果保留95%的信息
    transfer = PCA(n_components=0.95)
    data1 = transfer.fit_transform(tab4)
    print("数据形状：\n",data1.shape)
    
    return None

if __name__=="__main__":
    pca_demo()

'''PCA说明：一种提取特征的方法之一，是一种将复杂度高的数据尽可能降低为复杂程度低的
方法，但在此过程中会损失少量原有的数据信息并可能创造出新的数据变量，是将数据的特征
值以较小的损失集中到同一维度的过程'''