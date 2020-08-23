import numpy as np
from preprocessing import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split


class Mathai():
	def __init__(self):
		pass
	def fitpredict(self, X, y):		
		#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
		X_train, X_test, y_train, y_test = preprocessing(X, y)
		clf = RidgeClassifier().fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		print(accuracy_score(y_pred, y_test))
		
		

X = np.loadtxt('../dataset/train.csv')
y = np.loadtxt('../dataset/train_label.csv')
		
model = Mathai()
model.fitpredict(X, y)	