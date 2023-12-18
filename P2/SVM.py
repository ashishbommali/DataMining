import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

def add_all_row_and_column(matrix):
    # Calculate sums for rows and columns
    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)
    
    # Create a new matrix with an additional row and column for 'All'
    new_matrix = pd.DataFrame(matrix)
    new_matrix['All'] = row_sums
    new_matrix.loc['All'] = column_sums.tolist() + [row_sums.sum()]
    
    return new_matrix 

# Load the dataset
data = pd.read_csv("nba2021.csv")

selected_features = ['FG%', 'FT%', '3P%', 'TRB', 'AST', 'BLK', 'PTS', 'TOV', 'FGA']

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

# Create an SVM model
svm = SVC(kernel='linear', C=1.0, coef0=0.2, random_state=0, degree=3, shrinking=True, class_weight=None, tol=1e-3, max_iter=10000000)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)
accuracy = accuracy_score(y_test, svm_predictions)

print("Task 1")
train_predictions = svm.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Create a confusion matrix for SVM
confusion_matrix_svm = confusion_matrix(y_test, svm_predictions)
print("Task 2")
print("Confusion Matrix:")
print(add_all_row_and_column(confusion_matrix_svm))

print("-" * 150)

print("Task 3")

# Cross-validation with stratified k-fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
fold = 1
avg_accuracy_svm = 0

# Lists to store accuracy for each fold
train_accuracies = []
test_accuracies = []

# Lists to store confusion matrices for each fold
confusion_matrices = []

for train_idx, test_idx in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[test_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

    # Fit the SVM classifier on the training data
    svm.fit(X_train_fold, y_train_fold)

    # Test set predictions for SVM
    svm_predictions = svm.predict(X_val_fold)
    svm_test_accuracy = accuracy_score(y_val_fold, svm_predictions)
    test_accuracies.append(svm_test_accuracy)

    # Train set predictions for SVM
    train_svm_predictions = svm.predict(X_train_fold)
    svm_train_accuracy = accuracy_score(y_train_fold, train_svm_predictions)
    train_accuracies.append(svm_train_accuracy)

    # Create confusion matrix for SVM
    confusion_matrix_svm = confusion_matrix(y_val_fold, svm_predictions)
    confusion_matrices.append(confusion_matrix_svm)

    # Output results for this fold
    print(f"Fold {fold}:")
    
    # Print train and test accuracies
    print(f"Train set accuracy: {svm_train_accuracy:.3f}")
    print(f"Test set accuracy: {svm_test_accuracy:.3f}")

    # # Print confusion matrix
    # print("Confusion Matrix:")
    # print(add_all_row_and_column(confusion_matrix_svm))

    print("-" * 40)

    avg_accuracy_svm += svm_test_accuracy
    fold += 1

# Calculate average accuracy across all folds
avg_accuracy_svm /= 10
print(f"Average Accuracy Across All Folds: {avg_accuracy_svm:.4f}")
