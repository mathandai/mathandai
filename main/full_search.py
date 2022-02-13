from itertools import combinations
import copy

import numpy as np

from params import params


class Full_search:

    '''Реализация отбора признака полным перебором комбинаций признаков.'''

    def __init__(self, estimator_list=['SGDRegressor']):
        # определяем модель по которой отбираем признаки
        self.estimator_list = estimator_list

        # Cловарь {estimator : [Наименьшая ошибка, список признаков на которой достигается эта ошибке]}
        self.result_dict = {}

    def calcul_Q(self, subset_features):
        self.estimator.fit(self.X_train[:, subset_features], self.y_train)
        Y_pred = self.estimator.predict(self.X_test[:, subset_features])
        if self.task == 'Regression':
            return np.square(np.subtract(self.y_test, Y_pred)).mean()
        else:
            pass

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

            print(self.estimator)

            full_list = [i for i in range(X_train.shape[1])]

            for i in range(1, X_train.shape[1] + 1):
                for sub_features in combinations(full_list, i):

                    now_Q = self.calcul_Q(sub_features)
                    print(t, sub_features, now_Q)

                    if self.Q_best > now_Q:
                        self.Q_best = now_Q
                        self.best_sub_features = copy.deepcopy(sub_features)

            # Вернем значение наименьшей ошибки и список признаков на котором она достигается

            self.result_dict[estim] = [self.Q_best, self.best_sub_features]
