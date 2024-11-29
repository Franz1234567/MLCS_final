import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score

def preprocess(dataset, split_test = 0.2):
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

def tune_knn(list_dist, list_k):
	list_result = []
	for dist in list_dist:	
		for k in list_k:
			print("k:", k, "and dist:", dist)
			model = KNeighborsClassifier(n_neighbors=k, metric=dist)
			cv_scores = cross_val_score(model, np.array(X_train_val), np.array(y_train_val), cv=10, scoring="f1_macro")
			f1 = np.mean(cv_scores)
			cv_scores = cross_val_score(model, np.array(X_train_val), np.array(y_train_val), cv=10, scoring="accuracy")
			acc = np.mean(cv_scores)
			list_result.append({ "dist" : dist, "k": k, "f1": f1, "acc": acc})
	df_result = pd.DataFrame(list_result)
	return (df_result)
# cosine 6 -> f1_score: 0.9275 and acc: 0.9291

import warnings
from sklearn.exceptions import ConvergenceWarning

def tune_logistic(list_c, list_max_iter):
	list_result = []
	for c in list_c:	
		for iter in list_max_iter:
			print("c:",c, "and iter:", iter)
			model = LogisticRegression(C=c, max_iter=iter)
			cv_scores = cross_val_score(model, np.array(X_train_val), np.array(y_train_val), cv=10, scoring="f1_macro")
			f1 = np.mean(cv_scores)
			cv_scores = cross_val_score(model, np.array(X_train_val), np.array(y_train_val), cv=10, scoring="accuracy")
			acc = np.mean(cv_scores)
			list_result.append({ "c" : c, "max_iter": iter, "f1": f1, "acc": acc})
	df_result = pd.DataFrame(list_result)
	return (df_result)
# 10, 50 -> f1_score: 0.9608 and acc:0.9616

def tune_gradient_boost(n_estimator, learn_rate, max_depth):
	list_result = []
	for est in n_estimator:
		for rate in learn_rate:
			for depth in max_depth:
				print("n_estimator:", est, " learning_rate:", rate, "max_depth: ", depth)
				model = GradientBoostingClassifier(n_estimators=est, learning_rate=rate, max_depth=depth, random_state=42)
				cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=10, scoring='f1_macro')
				f1 = np.mean(cv_scores)
				cv_scores = cross_val_score(model, np.array(X_train_val), np.array(y_train_val), cv=10, scoring="accuracy")
				acc = np.mean(cv_scores)
				list_result.append({ "n_estimator" : est, "learning_rate": rate, "max_depth": depth, "f1": f1, "acc": acc})
	df_result = pd.DataFrame(list_result)
	return (df_result)
# 200 - 0.1 - 3 -> f1_score: 0.9611 and acc: 0.9619

def predict_train():

	print("KNN:")
	model = KNeighborsClassifier(n_neighbors=6, metric="cosine")
	model.fit(np.array(X_train_val), np.array(y_train_val))
	y_pred_train = model.predict(np.array(X_train_val))
	print("f1 score:", f1_score(y_train_val, y_pred_train, average="macro"))
	print("accuracy:", accuracy_score(y_train_val, y_pred_train))

	print("\nLogisticRegression:")
	model = LogisticRegression(C=10, max_iter=50)
	model.fit(X_train_val, y_train_val)
	y_pred_train = model.predict(X_train_val)
	print("f1 score:", f1_score(y_train_val, y_pred_train, average="macro"))
	print("accuracy:", accuracy_score(y_train_val, y_pred_train))

	print("\nGradient Boosting:")
	model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42)
	model.fit(X_train_val, y_train_val)
	y_pred_train = model.predict(X_train_val)
	print("f1 score:", f1_score(y_train_val, y_pred_train, average="macro"))
	print("accuracy:", accuracy_score(y_train_val, y_pred_train))



if __name__=="__main__":
	path_file = 'Android_Malware_Benign.csv'
	dataset = pd.read_csv(path_file)
	# randomize order of rows with a seed
	dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

	X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = preprocess(dataset)

	####################### uncomment to test knn parameters
	# df = tune_knn(['minkowski', 'cosine'], list(range(1,10)))
	# print(df.sort_values(by="f1", ascending=False).head(10))
	# print(df.sort_values(by="acc", ascending=False).head(10))

	####################### uncomment to test lr parameters
	######### Part1
	#warnings.filterwarnings("ignore", category=ConvergenceWarning) #avoiding visual pollution
	# list_c = [0.001, 0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000]
	# list_iter = [50, 100, 150, 200, 250, 300, 500, 1000, 2000, 5000]
	# df = tune_logistic(list_c, list_iter)
	# print(df.sort_values(by="f1", ascending=False).head(10))
	# print(df.sort_values(by="acc", ascending=False).head(10))	

	######### Part2: tuning parameters according to previous results
	# warnings.filterwarnings("ignore", category=ConvergenceWarning) #avoiding visual pollution
	# list_c = [5, 10, 15, 20, 25]
	# list_iter = [50, 100, 150, 200, 250, 300, 500, 1000]
	# df = tune_logistic(list_c, list_iter)
	# print(df.sort_values(by="f1", ascending=False).head(10))
	# print(df.sort_values(by="acc", ascending=False).head(10))

	####################### uncomment to test gradient boosting paramters
	# n_estimator = [50, 100, 200]
	# learn_rate = [0.01, 0.05, 0.1, 0.2, 0.3]
	# max_depth = [3, 4, 5]
	# df = tune_gradient_boost(n_estimator, learn_rate, max_depth)
	# print(df.sort_values(by="f1", ascending=False).head(10))
	# print(df.sort_values(by="acc", ascending=False).head(10))

	####################### uncomment to test the parameters found on the train_val set
	# predict_train()