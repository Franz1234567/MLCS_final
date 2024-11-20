import pandas as pd
import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning


def preprocess(dataset):
    dataset_prepprocess = dataset.copy()

    for column in dataset.columns:
        if dataset[column].nunique()== 1:
            dataset_prepprocess.drop(column, axis=1, inplace=True)

    X = dataset_prepprocess.drop('Label', axis=1)

    y = dataset_prepprocess['Label']
    y = y.replace('Malware', 1)
    y = y.replace('Benign', 0)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val

def test(model_name, train_features, train_labels, test_features, test_overfit=False):
    if model_name=="knn":
        model = KNeighborsClassifier(n_neighbors=6, metric="cosine")
    if model_name=="lr":
        model = LogisticRegression(C=10, max_iter=50)
    if model_name=="nb":
        model = naive_bayes.BernoulliNB()
    if model_name=="dt":
        model = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    if model_name=="rf":
        model = RandomForestClassifier(criterion="entropy", max_depth=10, n_estimators=200)
    if model_name=="svm":
        model = SVC(kernel="rbf", C=10, gamma=0.01)
    if model_name=="gb":
        model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
    time_before_fit = time.time()
    model.fit(np.array(train_features), np.array(train_labels))
    time_after_fit = time.time()
    time_taken = time_after_fit - time_before_fit
    print("Time taken to fit the model: ", time_taken)
    predicted_labels = model.predict(np.array(test_features))

    if test_overfit:
        train_predicted_labels = model.predict(np.array(train_features))
        print("Train f1 score: ", f1_score(train_labels, train_predicted_labels))
        print("Train accuracy: ", accuracy_score(train_labels, train_predicted_labels))
    return predicted_labels

def plot_confusion_matrix(y_true, y_pred, model_used):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, 
                xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
    
    plt.title('Confusion Matrix using ' + model_used)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    
    plt.show()
    

if __name__=="__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning) # avoiding visual pollution
    path_file = 'Android_Malware_Benign.csv'
    dataset = pd.read_csv(path_file)
	# randomize order of rows with a seed
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = preprocess(dataset)

    print("KNN:")   
    predict = test("knn", X_train_val, y_train_val, X_test, test_overfit=True)
    print("Test f1 score:", f1_score(y_test, predict))
    print("Test accuracy:", accuracy_score(y_test, predict))
    plot_confusion_matrix(y_test, predict, "KNN")

    print("\nNaive Bayes:")  
    predict = test("nb", X_train_val, y_train_val, X_test, test_overfit=True)
    print("Test f1 score:", f1_score(y_test, predict))
    print("Test accuracy:", accuracy_score(y_test, predict))
    plot_confusion_matrix(y_test, predict, "Naive Bayes")
    
    print("\nLogistic Regression:")
    predict = test("lr", X_train_val, y_train_val, X_test, test_overfit=True)
    print("Test f1 score:", f1_score(y_test, predict))
    print("Test accuracy:", accuracy_score(y_test, predict))
    plot_confusion_matrix(y_test, predict, "Logistic Regression")
    
    print("\nDecision Tree:")
    predict = test("dt", X_train_val, y_train_val, X_test, test_overfit=True)
    print("Test f1 score:", f1_score(y_test, predict))
    print("Test accuracy:", accuracy_score(y_test, predict))
    plot_confusion_matrix(y_test, predict, "Decision Tree")
    
    print("\nRandom Forest:")
    predict = test("rf", X_train_val, y_train_val, X_test, test_overfit=True)
    print("Test f1 score:", f1_score(y_test, predict))
    print("Test accuracy:", accuracy_score(y_test, predict))
    plot_confusion_matrix(y_test, predict, "Random Forest")
    
    print("\nSVM:")
    predict = test("svm", X_train_val, y_train_val, X_test, test_overfit=True)
    print("Test f1 score:", f1_score(y_test, predict))
    print("Test accuracy:", accuracy_score(y_test, predict))
    plot_confusion_matrix(y_test, predict, "SVM")

    print("\nGradient Boosting:")
    predict = test("gb", X_train_val, y_train_val, X_test, test_overfit=True)
    print("Test f1 score:", f1_score(y_test, predict))
    print("Test accuracy:", accuracy_score(y_test, predict))
    plot_confusion_matrix(y_test, predict, "Gradient Boosting")
    
    