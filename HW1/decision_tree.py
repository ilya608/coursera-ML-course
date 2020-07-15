import graphviz
import pandas as pd
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib as plt

def convertString(male):
    return 0 if male == 'male' else  1

features = ['Age', 'Sex', 'Fare', 'Pclass']
target = 'Survived'
df = pd.read_csv('titanic.csv')
df = df.loc[:, features + [target]]
# df = df[features + [target]]
df = df.dropna() # удалить строки с пропущенными значениями
X = df.loc[:, features]
Y = df.loc[:, target]
X.Sex = X.Sex.apply(convertString)
print(123)
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, Y)

importances = clf.feature_importances_
print(importances)

dot_data = tree.export_graphviz(clf, class_names=['0', '1'], out_file='res.txt')
graph = graphviz.Source(dot_data)

tree.plot_tree(clf, f)