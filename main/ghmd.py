import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from params import params
import copy


def mape(a, f):
    a = np.array(a)
    ind = np.where(a != 0)
    a = a[ind]
    f = f[ind]
    k = 100 / len(a)
    return k * np.sum(np.abs((a - f) / a))


class GMDH:
    """Метод группового учета аргументов (Group Method of Data Handling)"""

    def __init__(self, estimator_list=['SGDRegressor'], d_size=5, b_size=30):
        # определяем модель по которой отбираем признаки
        self.estimator_list = estimator_list

        # определяем сколько добавлять и удалять признков в 1 итерации
        self.d_size = d_size
        self.b_size = b_size

        # Cловарь {estimator : [Наименьшая ошибка, список признаков на которых достигается эта ошибке]}
        self.Result_dict = {}

    def calcul_Q(self, subset_features):
        self.estimator.fit(self.X_train[:, subset_features], self.y_train)
        Y_pred = self.estimator.predict(self.X_test[:, subset_features])
        if self.task == 'Regression':
            return (np.square(np.subtract(self.y_test, Y_pred)).mean()) ** 1 / 2
            # return np.divide(np.abs(np.subtract(self.y_test, Y_pred)), self.y_test).mean()
            # return mape(self.y_test, Y_pred)
        else:
            pass

    def feature_add(self, subset_features):
        Q = []
        features_list = []
        for r_ij in subset_features:
            for i in range(self.dim):
                if i not in r_ij:
                    temp_features = r_ij + [i]
                    temp_features.sort()
                    if temp_features not in features_list:
                        Q.append(self.calcul_Q(temp_features))
                        features_list.append(temp_features)
        features_list = list(zip(Q, features_list))
        features_list.sort()
        return [j for i, j in features_list[:self.b_size]], features_list[0][0]

    def fit(self, X_train, X_test, y_train, y_test, task='Regression'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.task = task
        self.dim = X_train.shape[1]

        for estim in self.estimator_list:
            self.estimator = params[self.task][estim]['estimator']

            self.Q_best = 1e12
            self.best_sub_features = []

            sub_features = [[]]
            history_Q = []

            local_Q_best = 1e12
            local_best_sub_features = []

            print(self.estimator)

            for j in range(self.dim):
                # добавили новый признак
                sub_features, now_Q = self.feature_add(sub_features)
                if now_Q < self.Q_best:
                    self.Q_best = now_Q
                    self.best_sub_features = sub_features[0]
                    j_0 = j
                print(sub_features)
                print(j, now_Q, sub_features[0])

                if j - j_0 > self.d_size:
                    self.Result_dict[estim] = [self.Q_best, self.best_sub_features]
                    break
