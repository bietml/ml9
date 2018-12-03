#import numpy as np
#import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import datasets
iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target
X_train,X_test,y_train,y_test=train_test_split(iris_data,iris_labels,test_size=0.20)
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
print("confusion matrix")
print(confusion_matrix(y_test,y_pred))
print("Accuracy matrix")
print(classification_report(y_test,y_pred))
