# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, classification_report, roc_curve, roc_auc_score

# Define file path
file_path = 'C:/Users/MADHAB MANNA/OneDrive/Desktop/Final lung cancer project/cancer analyze dataset.csv'

# Check if file exists
if os.path.exists(file_path):
    # Load dataset
    dataset = pd.read_csv(file_path)
    print(f"Dataset loaded with {len(dataset)} records")
    print(dataset.head())

    # Update result labels
    dataset['Result'] = dataset['Result'].replace({1: 'Cancer', 0: 'Non-Cancer'})

    # Bar chart for cancer and non-cancer percentages
    result_counts = dataset['Result'].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 4))
    plt.bar(result_counts.index, result_counts.values, color=['red', 'green'], edgecolor='black')
    plt.title("Cancer vs Non-Cancer Percentages", fontsize=14, weight='bold')
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.show()

    # Pairplot for feature distributions
    sns.pairplot(dataset.select_dtypes(include=np.number).join(dataset['Result']), hue="Result", diag_kind="kde", markers=["o", "s"], palette="coolwarm")
    plt.suptitle("Pairplot Analysis", y=1.02)
    plt.show()

    # Function for improved spline plots
    def plot_spline(x, y, hue, xlabel, ylabel, title, data, colors):
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, x=x, y=y, hue=hue, palette=colors, marker='o', linewidth=2)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, weight="bold")
        plt.legend(title=hue, fontsize=10)
        plt.grid(True)
        plt.show()

    # Spline plot for Age vs Smoke
    plot_spline("Age", "Smoke", "Result", "Age", "Smoke", "Smoke vs Age", dataset, colors={"Cancer": "red", "Non-Cancer": "green"})

    # Spline plot for Age vs Alcohol
    plot_spline("Age", "Alcohol", "Result", "Age", "Alcohol", "Alcohol vs Age", dataset, colors={"Cancer": "blue", "Non-Cancer": "orange"})

    # Additional figures for better analysis
    # Boxplot for Age distribution by Result with fixed palette and hue
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=dataset, x="Result", y="Age", hue="Result", palette={"Cancer": "red", "Non-Cancer": "green"}, legend=False)
    plt.title("Age Distribution by Result", fontsize=14, weight='bold')
    plt.xlabel("Result", fontsize=12)
    plt.ylabel("Age", fontsize=12)
    plt.show()


    # Correlation heatmap (numeric data only)
    plt.figure(figsize=(10, 8))
    numeric_dataset = dataset.select_dtypes(include=np.number)
    corr = numeric_dataset.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap", fontsize=14, weight='bold')
    plt.show()

    # Split dataset
    X = dataset.iloc[:, 3:5]
    Y = dataset['Result']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.2)

    # Train Random Forest Classifier
    classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    classifier.fit(X_train, Y_train)

    # Predictions and evaluation
    Y_pred = classifier.predict(X_test)

    cm = confusion_matrix(Y_test, Y_pred, labels=['Non-Cancer', 'Cancer'])
    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()
    print("True Negatives (Non-Cancer correctly classified):", tn)
    print("False Positives (Non-Cancer incorrectly classified):", fp)
    print("False Negatives (Cancer incorrectly classified):", fn)
    print("True Positives (Cancer correctly classified):", tp)

    print('F1 Score:', f1_score(Y_test, Y_pred, average="weighted") * 100)
    print('Accuracy:', accuracy_score(Y_test, Y_pred) * 100)
    print("Classification Report:\n", classification_report(Y_test, Y_pred))

    # ROC Curve for classification performance
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

else:
    print(f"Error: File not found at {file_path}")
