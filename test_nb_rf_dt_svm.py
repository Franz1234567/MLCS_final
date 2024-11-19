import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn import naive_bayes, neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

def preprocess(dataset):
    dataset_prepprocess = dataset.copy()

    for column in dataset.columns:
        if dataset[column].nunique()== 1:
            dataset_prepprocess.drop(column, axis=1, inplace=True) # removving column with only same value

    X = dataset_prepprocess.drop('Label', axis=1)

    y = dataset_prepprocess['Label']

    # print(y.value_counts())
    y = y.replace('Malware', 1)
    y = y.replace('Benign', 0)

    # split in train, test and validation in 60 20 20

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val

def test_model_NB(): # best one is BernoulliNB, see image
    for model in [naive_bayes.BernoulliNB(), naive_bayes.MultinomialNB()]:
        print(model)
        cv_scores_f1 = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='f1_macro')
        cv_scores_accuracy = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='accuracy')
        print('Cross-Validation f1_score:', np.mean(cv_scores_f1))
        print('Cross-Validation accuracy:', np.mean(cv_scores_accuracy))

def test_model_RF():
    results = []

    for criterion in ['gini', 'entropy', 'log_loss']:
        for max_depth in [None, 10, 20, 30]:
            for n_estimators in [10, 50, 100, 200]:
                
                model = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
                
                cv_scores_f1 = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='f1_macro')
                cv_scores_accuracy = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='accuracy')
                
                mean_f1 = np.mean(cv_scores_f1)
                mean_accuracy = np.mean(cv_scores_accuracy)
                
                results.append({
                    'criterion': criterion,
                    'max_depth': max_depth,
                    'n_estimators': n_estimators,
                    'mean_f1_score': mean_f1,
                    'mean_accuracy': mean_accuracy,
                })

    results_df = pd.DataFrame(results)

    top_f1 = results_df.sort_values(by='mean_f1_score', ascending=False).head(10)
    top_accuracy = results_df.sort_values(by='mean_accuracy', ascending=False).head(10)

    print("Top 10 results based on F1 score:")
    print(top_f1[['criterion', 'max_depth', 'n_estimators', 'mean_f1_score']])

    print("\nTop 10 results based on Accuracy:")
    print(top_accuracy[['criterion', 'max_depth', 'n_estimators', 'mean_accuracy']])

def test_model_dt():
    dt_results_list = []
    for criterion in ['gini', 'entropy']:
        for max_depth in [None, 10, 20, 30, 100, 200]:  
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
            
            cv_scores_f1 = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='f1_macro')
            cv_scores_accuracy = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='accuracy')
            
            mean_f1 = np.mean(cv_scores_f1)
            mean_accuracy = np.mean(cv_scores_accuracy)
            
            dt_results_list.append({
                'criterion': criterion,
                'max_depth': max_depth,
                'mean_f1_score': mean_f1,
                'mean_accuracy': mean_accuracy
            })

    dt_results_df = pd.DataFrame(dt_results_list)

    print("\nTop 10 results by F1 score:")
    print(dt_results_df.sort_values(by='mean_f1_score', ascending=False).head(10))

    print("\nTop 10 results by Accuracy:")
    print(dt_results_df.sort_values(by='mean_accuracy', ascending=False).head(10))

def test_model_svm():
    svm_results_list = []

    for kernel in ['linear', 'rbf', 'poly']:
        for C in [0.1, 1, 10, 100]:  
            for gamma in ['scale', 'auto', 0.01, 0.1, 1]:
                if kernel == 'linear' and gamma != 'scale':
                    continue  # Gamma not used in linear kernel

                model = SVC(kernel=kernel, C=C, gamma=gamma)
                
                cv_scores_f1 = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='f1_macro')
                cv_scores_accuracy = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='accuracy')
                
                mean_f1 = np.mean(cv_scores_f1)
                mean_accuracy = np.mean(cv_scores_accuracy)
                
                svm_results_list.append({
                    'kernel': kernel,
                    'C': C,
                    'gamma': gamma,
                    'mean_f1_score': mean_f1,
                    'mean_accuracy': mean_accuracy,
                })

    svm_results_df = pd.DataFrame(svm_results_list)

    # Display the top 10 configurations by F1 score and accuracy
    print("\nTop 10 results by F1 score:")
    print(svm_results_df.sort_values(by='mean_f1_score', ascending=False).head(10))

    print("\nTop 10 results by Accuracy:")
    print(svm_results_df.sort_values(by='mean_accuracy', ascending=False).head(10))



if __name__ == '__main__':
    path_file = 'Android_Malware_Benign.csv'
    dataset = pd.read_csv(path_file)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = preprocess(dataset)

    # test_model_NB()
    test_model_RF()
    # test_model_dt()
    # test_model_svm()