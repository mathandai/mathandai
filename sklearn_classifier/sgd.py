import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

X = np.loadtxt('../dataset/train.csv')
y = np.loadtxt('../dataset/train_label.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sgd_clf = SGDClassifier().fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

print(accuracy_score(y_pred, y_test))


