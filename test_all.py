import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

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

def test(model_name, train_features, train_labels, test_features):
    if model_name=="knn":
        model = KNeighborsClassifier(n_neighbors=6, metric="cosine")
    if model_name=="lr":
        model = LogisticRegression(C=10, max_iter=50)
    if model_name=="nb":
        model = naive_bayes.BernoulliNB()
    if model_name=="dt":
        model = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    if model_name=="rf":
        model = RandomForestClassifier(criterion="gini", max_depth=10, n_estimators=100)
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
    return predicted_labels

if __name__=="__main__":
    path_file = 'Android_Malware_Benign.csv'
    dataset = pd.read_csv(path_file)
	# randomize order of rows with a seed
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = preprocess(dataset)

    print("KNN:")   
    predict = test("knn", X_train_val, y_train_val, X_test)
    print("f1 score:", f1_score(y_test, predict))
    print("accuracy:", accuracy_score(y_test, predict))

    print("\nNaive Bayes:")  
    predict = test("nb", X_train_val, y_train_val, X_test)
    print("f1 score:", f1_score(y_test, predict))
    print("accuracy:", accuracy_score(y_test, predict))
    
    print("\nLogistic Regression:")
    predict = test("lr", X_train_val, y_train_val, X_test)
    print("f1 score:", f1_score(y_test, predict))
    print("accuracy:", accuracy_score(y_test, predict))
    
    print("\nDecision Tree:")
    predict = test("dt", X_train_val, y_train_val, X_test)
    print("f1 score:", f1_score(y_test, predict))
    print("accuracy:", accuracy_score(y_test, predict))
    
    print("\nRandom Forest:")
    predict = test("rf", X_train_val, y_train_val, X_test)
    print("f1 score:", f1_score(y_test, predict))
    print("accuracy:", accuracy_score(y_test, predict))
    
    print("\nSVM:")
    predict = test("svm", X_train_val, y_train_val, X_test)
    print("f1 score:", f1_score(y_test, predict))
    print("accuracy:", accuracy_score(y_test, predict))

    print("\nGradient Boosting:")
    predict = test("gb", X_train_val, y_train_val, X_test)
    print("f1 score:", f1_score(y_test, predict))
    print("accuracy:", accuracy_score(y_test, predict))
    
    