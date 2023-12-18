import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv("nba2021.csv")

# Drop columns that won't be used (e.g., Player and Tm)
data = data.drop(['Player', 'Tm'], axis=1)

# Split the data into features and target
X = data.drop('Pos', axis=1)
y = data['Pos']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=0)
rf_classifier.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Print feature importances
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")

# Select features with importance greater than a threshold
selected_features = X.columns[feature_importances > 0.035]


# Use only the selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Train a Random Forest classifier with selected features
rf_classifier_selected = RandomForestClassifier(random_state=0)
rf_classifier_selected.fit(X_train_selected, y_train)

# Make predictions and calculate accuracy
y_pred_selected = rf_classifier_selected.predict(X_test_selected)
test_accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"Test Accuracy (Random Forest with selected features): {test_accuracy_selected:.2f}")
