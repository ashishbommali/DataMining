import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("nba2021.csv")

selected_features = ['FG%', 'FT%', '3P%', 'TRB', 'AST', 'BLK',]
# selected_features = ['AST', 'FGA', 'MP', 'FG%',  '3P%', '2P%', 'eFG%',  'STL', 'BLK', 'TOV', 'PTS', 'TRB', 'FT%'] 

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

# Create a simple neural network model
mlp = MLPClassifier(
    hidden_layer_sizes=(5000, 1000), 
    max_iter=50000, 
    early_stopping=True, 
    learning_rate='adaptive', 
    random_state=0,
    )

mlp.fit(X_train,y_train)
mlp_predictions = mlp.predict(X_test)
accuracy = accuracy_score(y_test, mlp_predictions)
print("Task 1")
train_predictions = mlp.predict(X_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Cross-validation with stratified k-fold
kfold = StratifiedKFold(
    n_splits=10, 
    shuffle=True,  
    random_state=0)

fold = 1
avg_accuracy_mlp = 0

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

    # Fit the MLPClassifier on the training data
    mlp.fit(X_train_fold, y_train_fold)

    # Test set predictions for MLPClassifier
    mlp_predictions = mlp.predict(X_val_fold)
    mlp_test_accuracy = accuracy_score(y_val_fold, mlp_predictions)
    test_accuracies.append(mlp_test_accuracy)

    # Train set predictions for MLPClassifier
    train_mlp_predictions = mlp.predict(X_train_fold)
    mlp_train_accuracy = accuracy_score(y_train_fold, train_mlp_predictions)
    train_accuracies.append(mlp_train_accuracy)

    # Create confusion matrix for MLPClassifier
    confusion_matrix_mlp = confusion_matrix(y_val_fold, mlp_predictions)
    confusion_matrices.append(confusion_matrix_mlp)

    # Output results for this fold
    print(f"Fold {fold}:")

    print(f"Train Accuracy: {mlp_train_accuracy}")
    print(f"Test Accuracy: {mlp_test_accuracy}")

    confusion_matrix_dtree = add_all_row_and_column(confusion_matrix_mlp)
    print("Confusion Matrix:")
    print(pd.DataFrame(confusion_matrix_mlp))

    print("-" * 40)

    avg_accuracy_mlp += mlp_test_accuracy
    fold += 1

# Calculate average accuracy across all folds
avg_accuracy_mlp /= 10
print(f"Average Accuracy Across All Folds: {avg_accuracy_mlp:.4f}")
