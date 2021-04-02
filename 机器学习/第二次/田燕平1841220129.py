from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

news=fetch_20newsgroups()
#获取数据 fetch_20news
x_train,x_text,y_train,y_text= train_test_split(news.data, news.target, random_state=6)
#划分数据集
vectorizer = TfidfVectorizer()
x_train= vectorizer.fit_transform(x_train)
print(vectorizer.get_feature_names())
#特征工程
model= MultinomialNB(alpha=1.0)
model.fit(x_train, y_train)
#贝叶斯预估器
x_test = vectorizer.transform(x_text)
y_predict = model.predict(x_test)
print("预测每篇文章的类别：", y_predict[:100])
print("真实类别为：", y_text[:100])
print("预测准确率为：", model.score(x_test, y_text))
#模型估计
