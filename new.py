import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree

print("Dataset:")
dataset = pd.read_csv('survey lung cancer.csv')
print(len(dataset))
print(dataset.head())

scatter_matrix(dataset)
pyplot.show()

A = dataset[dataset.Result == 1]
B = dataset[dataset.Result == 0]

plt.scatter(A.Age, A.Smoke, color="Red", label="1", alpha=0.4)
plt.scatter(B.Age, B.Smoke, color="Green", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Smoke")
plt.legend()
plt.title("Smoke vs Age")
plt.show()

plt.scatter(A.Age, A.Alcohol, color="pink", label="1", alpha=0.4)
plt.scatter(B.Age, B.Alcohol, color="black", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Alcohol vs Age")
plt.show()

plt.scatter(A.Smoke, A.Alcohol, color="yellow", label="1", alpha=0.4)
plt.scatter(B.Smoke, B.Alcohol, color="blue", label="0", alpha=0.4)
plt.xlabel("Smoke")
plt.ylabel("Alcohol")
plt.legend()
plt.title("Smoke vs Alcohol")
plt.show()





















# Splitting dataset
X = dataset.iloc[:, 3:5]
Y = dataset.iloc[:, 6]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

# Feature Scaling
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

print('------------------***Using KNN Algorithm***---------------')

import math
k = int(math.sqrt(len(Y_train)))
if k % 2 == 0:  
    k += 1
print("Optimal number of neighbors:", k)
classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')


classifier.fit(X_train, Y_train)


Y_pred = classifier.predict(X_test)
print(Y_pred)


cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])

print("Confusion Matrix: ")
print(cm)

tn, fp, fn, tp = cm.ravel()
print("True Negatives (Non-cancer correctly classified):", tn)
print("False Positives (Non-cancer incorrectly classified):", fp)
print("False Negatives (Cancer incorrectly classified):", fn)
print("True Positives (Cancer correctly classified):", tp)

print('F1 Score:', f1_score(Y_test, Y_pred, zero_division=1) * 100)

print('ACCURACY : ', accuracy_score(Y_test, Y_pred) * 100)

print("----------------********Using Decision Tree Algorithm*****----------------")


c = tree.DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=0)

# Fitting Decision Tree model
c.fit(X_train, Y_train)

# Calculating accuracy
accu_train = c.score(X_train, Y_train)
accu_test = c.score(X_test, Y_test)

print("--------*****Using Decision Tree****--------")
print('Classification accuracy on train:', accu_train * 100)
print('Classification accuracy on test:', accu_test * 100)
