import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
from sklearn.metrics import accuracy_score, confusion_matrix

random_seed = 42

class Task1SVM:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.selected_features = ['FG%','FT%','3P%','2P%', 'TRB', 'AST', 'BLK', 'PTS', 'TOV', 'FGA']
        self.X = self.data[self.selected_features]
        self.y = self.data['Pos']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25)
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        # self.svm =  SVC(kernel='linear', C=1.0, coef0=0.2, random_state=random_seed, degree=3, shrinking=True, probability=True, break_ties=True, class_weight=None, tol=1e-3, max_iter=100000)
        self.svm =  LinearSVC(
            penalty="l1", 
            C=1.0, 
            loss="squared_hinge", 
            dual=False, 
            fit_intercept=True, 
            tol=1e-1, 
            intercept_scaling=1.1, 
            multi_class='ovr', 
            class_weight=None, 
            max_iter=10000)

    def train_and_test(self):
        self.svm.fit(self.X_train, self.y_train)
        y_pred_train = self.svm.predict(self.X_train)
        y_pred_test = self.svm.predict(self.X_test)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        
        return train_accuracy, test_accuracy  

class Task2SVM(Task1SVM):
    def confusion_matrix(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25)
        self.svm.fit(X_train, y_train)
        y_pred_test = self.svm.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test)
        cm = self.add_mapping_to_confusion_matrix(cm)        
        
        return cm 
   
    def add_mapping_to_confusion_matrix(self, cm):
        num_classes = cm.shape[0]
        mapped_cm = np.zeros((num_classes , num_classes), dtype=int)
        mapped_cm[:num_classes, :num_classes] = cm

        new_matrix = pd.DataFrame(mapped_cm)             
        new_matrix['All'] = new_matrix.sum(axis=1)
        new_matrix.loc['All'] = new_matrix.sum(axis=0)     
        
        return new_matrix

class Task3SVM(Task1SVM):
    
    def cross_validation(self):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)

        # Perform cross-validation and calculate accuracy for each fold
        scores = cross_val_score(self.svm,self.X, self.y, cv=10)

        print("Accuracy for Each Fold:")
        for i, score in enumerate(scores):
            print("Fold {} Test_Accuracy: {:.2f}%".format(i + 1, score * 100))

        average_accuracy = scores.mean()
        print("Average Test Accuracy Across All Folds: {:.2f}%".format(average_accuracy * 100))

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
        accuracies = []
        
        for train_idx, test_idx in kfold.split(self.X, self.y):
            X_train_fold, X_test_fold = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train_fold, y_test_fold = self.y.iloc[train_idx], self.y.iloc[test_idx]
           
            self.svm.fit(X_train_fold, y_train_fold)
            y_pred_test = self.svm.predict(X_test_fold)
            fold_accuracy = accuracy_score(y_test_fold, y_pred_test)
            accuracies.append(fold_accuracy)

        return accuracies, np.mean(accuracies)

class MainClass:
    def __init__(self, data_file):
        self.data_file = data_file

    def task1(self):
        print("Task 1:")
        task1 = Task1SVM(self.data_file)   
        train_acc, test_acc = task1.train_and_test()
        print(f"Training Accuracy: {train_acc*100:.4f}")
        print(f"Test Accuracy: {test_acc*100:.4f}")
        print("-" * 40)

    def task2(self):
        print("Task 2:")
        task2 = Task2SVM(self.data_file)  
        cm = task2.confusion_matrix()
        print("Confusion Matrix:")
        print(cm)
        print("-" * 40)

    def task3(self):
        print("Task 3:")
        task3 = Task3SVM(self.data_file)  
        accuracies, avg_accuracy = task3.cross_validation()
        print("-" * 40)

if __name__ == "__main__":
    data_file = "nba2021.csv"
    np.random.seed(random_seed)
    main_class_obj = MainClass(data_file)
    main_class_obj.task1()
    main_class_obj.task2()
    main_class_obj.task3()
