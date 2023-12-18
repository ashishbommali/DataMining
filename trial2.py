import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Task 1: Classification using SVM
# Load the dataset
data = pd.read_csv("nba2021.csv")

# Remove irrelevant columns that are unlikely to contribute to predicting position
removed_columns = ["Player", "Tm", "Age", "MP", 'GS','FG' , '2P', '3P', 'FT'] 
data = data.drop(removed_columns, axis=1)

# Filter out players who played fewer than a certain number of games
min_games_threshold = 20  # You can adjust this threshold as needed
data = data[data["G"] >= min_games_threshold]
data = data.drop('G',axis=1)
# print(data)
# Define features and targets
X = data.iloc[:, 2:]  # Features (player stats, excluding irrelevant attributes)
y = data["Pos"]  # Target variable (positions)

# Split the data into training and testing sets (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Create and train the Support Vector Machine (SVM) classifier with regularization
svm = SVC(C=0.1, kernel='linear', degree=2)  # Adjust the value of C for regularization

svm.fit(X_train, y_train)

# Make predictions on the test set using the best model
y_pred = svm.predict(X_test)

# Calculate accuracy on the training and test sets
train_accuracy = svm.score(X_train, y_train)
test_accuracy = svm.score(X_test, y_test)

print("Training Set Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Set Accuracy: {:.2f}%".format(test_accuracy * 100))

# Task 2: Confusion Matrix
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the confusion matrix in the specified format
conf_matrix_df = pd.DataFrame(conf_matrix, columns=svm.classes_, index=svm.classes_)
conf_matrix_df.loc["Total"] = conf_matrix_df.sum()
conf_matrix_df["Total"] = conf_matrix_df.sum(axis=1)
print("Confusion Matrix:")
print(conf_matrix_df)

# Task 3: 10-fold Cross-Validation
# Create a 10-fold stratified cross-validation splitter
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Perform cross-validation and calculate accuracy for each fold
cv_scores = cross_val_score(svm, X, y, cv=cv)
print("Accuracy for Each Fold:")
for i, score in enumerate(cv_scores):
    print("Fold {}: {:.2f}%".format(i + 1, score * 100))

average_accuracy = cv_scores.mean()
print("Average Accuracy Across All Folds: {:.2f}%".format(average_accuracy * 100))
