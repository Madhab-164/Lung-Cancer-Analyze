import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report

print("Dataset:")
dataset = pd.read_csv('survey lung cancer.csv')
print(len(dataset))
print(dataset.head())

# Update result labels
dataset['Result'] = dataset['Result'].replace({1: 'Cancer', 0: 'Non-Cancer'})

# Improved scatter matrix visualization
sns.pairplot(dataset, hue="Result", diag_kind="kde", markers=["o", "s"], palette="coolwarm")
plt.suptitle("Pairplot Analysis", y=1.02)
plt.show()

# Improved graphics for visualizations
sns.set_theme(style="whitegrid")
def plot_graph(x, y, hue, xlabel, ylabel, title, data, colors):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=colors, s=100, alpha=0.7, edgecolor="black")
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.legend(title=hue, fontsize=10)
    plt.show()

plot_graph("Age", "Smoke", "Result", "Age", "Smoke", "Smoke vs Age", dataset, colors={"Cancer": "red", "Non-Cancer": "green"})
plot_graph("Age", "Alcohol", "Result", "Age", "Alcohol", "Alcohol vs Age", dataset, colors={"Cancer": "blue", "Non-Cancer": "orange"})
plot_graph("Smoke", "Alcohol", "Result", "Smoke", "Alcohol", "Smoke vs Alcohol", dataset, colors={"Cancer": "purple", "Non-Cancer": "cyan"})

# Splitting dataset
X = dataset.iloc[:, 3:5]
Y = dataset['Result']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

# # Feature Scaling
# sc_x = StandardScaler()
# X_train = sc_x.fit_transform(X_train)
# X_test = sc_x.transform(X_test)

print('------------------***Using Random Forest Algorithm***---------------')

# Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

# Metrics
cm = confusion_matrix(Y_test, Y_pred, labels=['Non-Cancer', 'Cancer'])
print("Confusion Matrix: ")
print(cm)

tn, fp, fn, tp = cm.ravel()
print("True Negatives (Non-Cancer correctly classified):", tn)
print("False Positives (Non-Cancer incorrectly classified):", fp)
print("False Negatives (Cancer incorrectly classified):", fn)
print("True Positives (Cancer correctly classified):", tp)

print('F1 Score:', f1_score(Y_test, Y_pred, average="weighted") * 100)
print('Accuracy:', accuracy_score(Y_test, Y_pred) * 100)
print("Classification Report:\n", classification_report(Y_test, Y_pred))

# Feature Importances
importances = classifier.feature_importances_
feature_names = X.columns
plt.figure(figsize=(8, 5))
plt.barh(feature_names, importances, color="skyblue", edgecolor="black")
plt.xlabel("Importance", fontsize=12)
plt.title("Feature Importances", fontsize=14, weight="bold")
plt.show()

# Advanced feature: Predict probabilities and ROC Curve
from sklearn.metrics import roc_curve, roc_auc_score

Y_prob = classifier.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(Y_test.map({'Non-Cancer': 0, 'Cancer': 1}), Y_prob)

fpr, tpr, thresholds = roc_curve(Y_test.map({'Non-Cancer': 0, 'Cancer': 1}), Y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Receiver Operating Characteristic", fontsize=14, weight="bold")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
