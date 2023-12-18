import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("nba2021.csv")

# Feature selection from random forest's feature importance which are > 0.03
# selected_features = ['AST', 'FGA', 'MP', 'FG%',  '3P%', '2P%', 'eFG%',  'STL', 'BLK', 'TOV', 'PTS', 'TRB', 'FT%'] 
selected_features = ['FG%', 'FT%', '3P%', 'TRB', 'AST', 'BLK',]


# Filter the dataset to include only the selected features
data = data[selected_features + ['Pos']]

# # Perform outlier detection and handling (removing rows containing outliers)
# Q1 = data[selected_features].quantile(0.25)
# Q3 = data[selected_features].quantile(0.75)
# IQR = Q3 - Q1
# data = data[~((data[selected_features] < (Q1 - 1.5 * IQR)) | (data[selected_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Split the data into features (X) and target (y)
X = data.drop('Pos', axis=1)
y = data['Pos']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

best_k = None
best_accuracy = 0

# Lists to store train and test accuracies for each fold
train_accuracies = []
test_accuracies = []

# # Create StandardScaler instances for both training and test sets
# scaler_train = StandardScaler()
# scaler_test = StandardScaler()

# # Fit the scaler on the training data and transform both training and test data
# X_train = scaler_train.fit_transform(X_train)
# X_test = scaler_train.transform(X_test)

# Iterate through different values of k
for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan', weights='distance')  # Define the KNN classifier
    knn.fit(X_train, y_train)
    
    # Perform cross-validation
    train_scores = cross_val_score(knn, X_train, y_train, cv=10)
    mean_train_accuracy = train_scores.mean()
    train_accuracies.append(mean_train_accuracy)
    
    test_accuracy = knn.score(X_test, y_test)
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_k = k

print(f"Best k: {best_k}")

# Initialize the KNN classifier with the best k
knn = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan', weights='distance')

# Train the KNN classifier
knn.fit(X_train, y_train)

# Test set predictions
predictions = knn.predict(X_test)
print("Test set predictions:\n", predictions)

# Test set accuracy
test_accuracy = knn.score(X_test, y_test)
print("Test set accuracy: {:.2f}".format(test_accuracy))

# Train set accuracy
train_accuracy = knn.score(X_train, y_train)
print("Train set accuracy: {:.2f}".format(train_accuracy))

# Display train accuracies for each fold
print("Train set accuracies for each fold:", train_accuracies)

# Perform cross-validation
scores = cross_val_score(knn, X, y, cv=10)
print("Test set accuracy for each fold (KNN):", scores)
print("Average Test Accuracy (KNN): {:.2f}".format(scores.mean()))
