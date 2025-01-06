import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, f1_score, accuracy_score, 
                             precision_score, recall_score, roc_auc_score)
import joblib
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Dataset
logging.info("Loading dataset...")
file_path = 'survey lung cancer.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset file '{file_path}' not found.")

dataset = pd.read_csv(file_path)
logging.info(f"Dataset loaded successfully with {len(dataset)} records.")
logging.info(f"First 5 records:\n{dataset.head()}")

# Check necessary columns
required_columns = ['Age', 'Smoke', 'Alcohol', 'Result']
missing_cols = [col for col in required_columns if col not in dataset.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in dataset: {missing_cols}")

# Data Visualization with Line Graphs
def plot_line(x, y, xlabel, ylabel, title):
   
    A = dataset[dataset.Result == 1]
    B = dataset[dataset.Result == 0]
    
    avg_A = A.groupby(x)[y].mean()
    avg_B = B.groupby(x)[y].mean()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=avg_A, label='Cancer', marker='o', color='red')
    sns.lineplot(data=avg_B, label='Non-cancer', marker='o', color='green')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Generate Line Graphs
plot_line('Age', 'Smoke', 'Age Groups', 'Average Smoke', 'Smoking Trends by Age')
plot_line('Age', 'Alcohol', 'Age Groups', 'Average Alcohol', 'Alcohol Consumption Trends by Age')
plot_line('Smoke', 'Alcohol', 'Smoke Levels', 'Average Alcohol', 'Alcohol Consumption by Smoke Levels')

# Data Splitting
logging.info("Splitting dataset...")
X = dataset[['Age', 'Smoke']]
Y = dataset['Result']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Feature Scaling
logging.info("Scaling features...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scalerr (Result = 1)
joblib.dump(scaler, 'scaler.pkl')

# KNN Algorithm
logging.info("Training KNN classifier...")
k = max(3, int(np.sqrt(len(Y_train))) | 1)  # Ensure odd k
knn_classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
knn_classifier.fit(X_train, Y_train)

# Save the KNN model
joblib.dump(knn_classifier, 'knn_model.pkl')

# Evaluate KNN
logging.info("Evaluating KNN classifier...")
Y_pred_knn = knn_classifier.predict(X_test)
logging.info(f"Confusion Matrix:\n{confusion_matrix(Y_test, Y_pred_knn)}")
logging.info(f"Accuracy: {accuracy_score(Y_test, Y_pred_knn) * 100:.2f}%")
logging.info(f"F1 Score: {f1_score(Y_test, Y_pred_knn) * 100:.2f}%")
logging.info(f"Precision: {precision_score(Y_test, Y_pred_knn) * 100:.2f}%")
logging.info(f"Recall: {recall_score(Y_test, Y_pred_knn) * 100:.2f}%")
logging.info(f"ROC-AUC: {roc_auc_score(Y_test, Y_pred_knn):.2f}")

# Decision Tree Algorithm
logging.info("Training Decision Tree classifier...")
tree_classifier = DecisionTreeClassifier(random_state=42)
tree_classifier.fit(X_train, Y_train)

# Save the Decision Tree model
joblib.dump(tree_classifier, 'decision_tree_model.pkl')

# Evaluate Decision Tree
logging.info("Evaluating Decision Tree classifier...")
train_accuracy = tree_classifier.score(X_train, Y_train) * 100
test_accuracy = tree_classifier.score(X_test, Y_test) * 100
logging.info(f"Train Accuracy: {train_accuracy:.2f}%")
logging.info(f"Test Accuracy: {test_accuracy:.2f}%")
