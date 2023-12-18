import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# Load the dataset
data = pd.read_csv("nba2021.csv")

# Drop columns that won't be used (e.g., Player and Tm)
data = data.drop(['Player', 'Tm'], axis=1)

# Split the data into features (X) and target (y)
X = data.drop('Pos', axis=1)
y = data['Pos']

# Feature selection from Random Forest's feature importance which are > 0.35
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(data.drop('Pos', axis=1), data['Pos'])

# Extract feature importances and selected features
importances = rf.feature_importances_
# Get feature importances
feature_importances = rf.feature_importances_

# Print feature importances
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

# Select features with importance greater than a threshold
selected_features = X.columns[feature_importances > 0.035]

# Filter the dataset to include only the selected features
data = data[selected_features]


# Split the data into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Decision Tree
dtree = DecisionTreeClassifier(max_depth=10, random_state=0)

# Cross-validation with stratified k-fold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
fold = 1
avg_accuracy_dtree = 0

# Lists to store accuracy for each fold
train_accuracies = []
test_accuracies = []

# Lists to store confusion matrices for each fold
confusion_matrices = []

def add_all_row_and_column(matrix):
    # Calculate sums for rows and columns
    row_sums = matrix.sum(axis=1)
    column_sums = matrix.sum(axis=0)
    
    # Create a new matrix with an additional row and column for 'All'
    new_matrix = pd.DataFrame(matrix)
    new_matrix['All'] = row_sums
    new_matrix.loc['All'] = column_sums.tolist() + [row_sums.sum()]
    
    return new_matrix

for train_idx, test_idx in kfold.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[test_idx]
    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[test_idx]

    # Fit the Decision Tree on the training data
    dtree.fit(X_train_fold, y_train_fold)

    # Test set predictions for Decision Tree
    dtree_predictions = dtree.predict(X_val_fold)
    dtree_test_accuracy = accuracy_score(y_val_fold, dtree_predictions)
    test_accuracies.append(dtree_test_accuracy)

    # Train set predictions for Decision Tree
    train_dtree_predictions = dtree.predict(X_train_fold)
    dtree_train_accuracy = accuracy_score(y_train_fold, train_dtree_predictions)
    train_accuracies.append(dtree_train_accuracy)

    # Create confusion matrix for Decision Tree
    confusion_matrix_dtree = confusion_matrix(y_val_fold, dtree_predictions)
    confusion_matrices.append(confusion_matrix_dtree)

    # Output results for this fold
    print(f"Fold {fold}:")
    
    print(f"Train Accuracy: {dtree_train_accuracy}")
    print(f"Test Accuracy: {dtree_test_accuracy}")

    confusion_matrix_dtree = add_all_row_and_column(confusion_matrix_dtree)
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix_dtree))

    print("-" * 40)

    avg_accuracy_dtree += dtree_test_accuracy
    fold += 1

# Calculate average accuracy across all folds
avg_accuracy_dtree /= 10
print(f"Average Accuracy Across All Folds: {avg_accuracy_dtree:.4f}")

