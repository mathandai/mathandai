import numpy as np
from params import params
from preprocessing import fillna, tt_split, scale, task

from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV


class Mathai:
	def __init__(self):
		pass

	def hyperopt(self):
		pass

	def fitpredict(self, X, y):
		# print(X)
		type_task = task(y)
		param_models = {'Classification':{'SVM':{}, 'Linear':{}}, 'Regression':{}}
		if np.isnan(X).any():
			inds = np.where(np.isnan(X))
			print(inds)
			for i in ['mean', 'zero']:
				X = fillna(X, inds, i)
				print(X)
				X_train, X_test, y_train, y_test = tt_split(X, y)
				X_train = scale(X_train)
				X_test = scale(X_test)
				print(X_train, y_train)
				print(X_test, y_test)
				# X_train, X_test, y_train, y_test = preprocessing(X, y)
				#svc = svm.SVC()
				mod = param_models[type_task]
				par = param_models[type_task]
				clf = GridSearchCV(mod, par)
				#clf = RidgeClassifier().fit(X_train, y_train)
				y_pred = clf.predict(X_test)
				print(accuracy_score(y_pred, y_test))
		else:
			X_train, X_test, y_train, y_test = tt_split(X, y)
			clf = RidgeClassifier().fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			print(accuracy_score(y_pred, y_test))
			X_train = scale(X_train)
			X_test = scale(X_test)
			clf = RidgeClassifier().fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			print(accuracy_score(y_pred, y_test))
		

X = np.loadtxt('../dataset/train.csv')
y = np.loadtxt('../dataset/train_label.csv')
# X = np.genfromtxt('../dataset/ds.csv', delimiter=';')
# y = np.loadtxt('../dataset/y.csv')
		
model = Mathai()
model.fitpredict(X, y)