import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Load the dataset
data = pd.read_csv("nba2021.csv")

selected_features = ['FG%', 'FT%', '3P%', 'TRB', 'AST', 'BLK']

# Filter the dataset to include only the selected features
data = data[selected_features + ['Pos']]

# Split the data into features (X) and target (y)
X = data.drop('Pos', axis=1)
y = data['Pos']

# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Naive Bayes classifier
gnb = GaussianNB(var_smoothing=1e-9)

# Cross-validation with stratified k-fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
fold = 1
avg_accuracy_gnb = 0

# Lists to store accuracy for each fold
train_accuracies = []
test_accuracies = []

# Lists to store confusion matrices for each fold
confusion_matrices = []

for train_idx, test_idx in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[test_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

    # Fit the Gaussian Naive Bayes classifier on the training data
    gnb.fit(X_train_fold, y_train_fold)

    # Test set predictions for Gaussian Naive Bayes
    gnb_predictions = gnb.predict(X_val_fold)
    gnb_test_accuracy = accuracy_score(y_val_fold, gnb_predictions)
    test_accuracies.append(gnb_test_accuracy)

    # Train set predictions for Gaussian Naive Bayes
    train_gnb_predictions = gnb.predict(X_train_fold)
    gnb_train_accuracy = accuracy_score(y_train_fold, train_gnb_predictions)
    train_accuracies.append(gnb_train_accuracy)

    # Create confusion matrix for Gaussian Naive Bayes
    confusion_matrix_gnb = confusion_matrix(y_val_fold, gnb_predictions)
    confusion_matrices.append(confusion_matrix_gnb)

    # Output results for this fold
    print(f"Fold {fold}:")

    print(f"Train Accuracy: {gnb_train_accuracy}")
    print(f"Test Accuracy: {gnb_test_accuracy}")
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix_gnb))

    print("-" * 40)

    avg_accuracy_gnb += gnb_test_accuracy
    fold += 1

# Calculate average accuracy across all folds
avg_accuracy_gnb /= 10
print(f"Average Accuracy Across All Folds: {avg_accuracy_gnb:.4f}")
