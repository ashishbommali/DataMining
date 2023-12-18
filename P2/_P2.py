import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("nba2021.csv")

# Random state for reproducibility
random_state = 0

# Drop 'Player' and 'Tm' columns
data = data.drop(['Player', 'Tm'], axis=1)

# Use label encoding for 'Pos'
label_encoder = LabelEncoder()
data['Pos'] = label_encoder.fit_transform(data['Pos'])

# Split the data 75% train and 25% test
X = data.drop('Pos', axis=1)
y = data['Pos']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

# Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=random_state)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
test_accuracy_tree = accuracy_score(y_test, y_pred_tree)

# Cross-validation - 10 fold
tree_cv_scores = cross_val_score(tree, X, y, cv=10)

# Random Forest Classifier
rf = RandomForestClassifier(random_state=random_state)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
test_accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Cross-validation - 10 fold
rf_cv_scores = cross_val_score(rf, X, y, cv=10)

# Display feature importances
feature_importances = rf.feature_importances_
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

# Confusion Matrix for Decision Tree
confusion_tree = confusion_matrix(y_test, y_pred_tree)

# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, 40, 50, None]
}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X, y)
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
test_accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)

# Confusion Matrix for Random Forest
confusion_rf = confusion_matrix(y_test, y_pred_rf)
confusion_best_rf = confusion_matrix(y_test, y_pred_best_rf)

# Results
print(f"Test Accuracy (Decision Tree): {test_accuracy_tree:.2f}")
print("Accuracy for each fold (Decision Tree):", tree_cv_scores)
print(f"Average Accuracy (Decision Tree): {tree_cv_scores.mean():.2f}")
print(f"Test Accuracy (Random Forest): {test_accuracy_rf:.2f}")
print("Accuracy for each fold (Random Forest):", rf_cv_scores)
print(f"Average Accuracy (Random Forest): {rf_cv_scores.mean():.2f}")
print("Confusion Matrix (Decision Tree):")
print(confusion_tree)
print(f"Test Accuracy (Tuned Random Forest): {test_accuracy_best_rf:.2f}")
print("Confusion Matrix (Random Forest):")
print(confusion_rf)
print("Confusion Matrix (Tuned Random Forest):")
print(confusion_best_rf)
