import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,precision_recall_fscore_support
df = pd.read_csv('Data.csv')
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
x_train,x_test,y_train,y_test = train_test_split(df[['Age','Experience','Rank','Nationality']],df['Go'],test_size=0.33)
clf = DecisionTreeClassifier()
clf.fit(np.array(x_train),np.array(y_train))
pred = clf.predict(np.array(x_test))
print(df.head())

print("Accuracy Score {}".format(accuracy_score(y_test,pred)))
print('precision_recall_fscore_support {}'.format(precision_recall_fscore_support(y_test,pred)))
print('Confusion Matrix {}'.format(confusion_matrix(y_test,pred)))

tree.plot_tree(clf,feature_names = ['Age', 'Experience', 'Rank', 'Nationality'])
plt.show()