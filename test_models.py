import numpy as np

from main.model import Mathai
from time import perf_counter
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split

X = np.loadtxt('./dataset/train.csv')
y = np.loadtxt('./dataset/train_label.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def timeit(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        rv = func(*args, **kwargs)
        print(perf_counter() - start)
        return rv
    return wrapper

#@timeit
sk_clf = RidgeClassifier().fit(X_train, y_train)
#@timeit
cat_clf = CatBoostClassifier(verbose=False).fit(X_train, y_train)
#@timeit
math_clf = Mathai()
#print(dir(math_clf))

y_pred_sk = sk_clf.predict(X_test)
y_pred_cat = cat_clf.predict(X_test)

print(accuracy_score(y_pred_sk, y_test), accuracy_score(y_pred_cat, y_test))

