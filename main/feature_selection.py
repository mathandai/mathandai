import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import copy


class Add_Del:
    def __init__(self, estimator=LinearRegression(), d_add=2, d_dell=1):
        # определяем модель по которой отбираем признаки
        self.estimator = estimator

        # определяем сколько добавлять и удалять признков в 1 итерации
        self.d_add = d_add
        self.d_dell = d_dell

    def calcul_Q(self, subset_features):
        self.estimator.fit(self.X_train[:, subset_features], self.y_train)
        Y_pred = self.estimator.predict(self.X_test[:, subset_features])
        return np.square(np.subtract(self.y_test, Y_pred)).mean()

    def feature_add(self, subset_features):
        Q = np.array([-1, 1e12])
        for i in range(self.dim):
            if i not in subset_features:
                g = np.array([i, self.calcul_Q(subset_features + [i])])
                Q = np.vstack((Q, g))
        y = Q[:, 1].min()
        return subset_features + [int(i) for i, j in Q if j == y]

    def feature_del(self, subset_features):
        Q = np.array([-1, 0])
        for j in subset_features:
            list_sub = copy.deepcopy(subset_features)
            list_sub.remove(j)
            g = np.array([j, self.calcul_Q(list_sub)])
            Q = np.vstack((Q, g))
        y = Q[:, 1].max()
        c = [i for i, j in Q if j == y]
        subset_features.remove(c[0])
        return subset_features

    def fit(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.dim = X_train.shape[1]
        t = 0

        self.Q_best = 1e12
        self.best_sub_features = []

        sub_features = []
        history_Q = []

        local_Q_best = 1e12
        local_best_sub_features = []

        while True:
            if self.d_add > 0:
                t_0 = t
                while len(sub_features) < self.dim:
                    t += 1

                    # добавили новый признак
                    sub_features = self.feature_add(sub_features)
                    now_Q = self.calcul_Q(sub_features)
                    history_Q.append(now_Q)

                    if local_Q_best > now_Q:
                        local_Q_best = now_Q
                        local_best_sub_features = copy.deepcopy(sub_features)
                        t_0_add = t

                    if t - t_0 >= self.d_add:
                        break

            if self.d_dell > 0:
                if local_best_sub_features:
                    sub_features = copy.deepcopy(local_best_sub_features)
                else:
                    sub_features = [k for k in range(self.dim)]

                t_0 = t
                while len(sub_features) > 0:
                    t += 1

                    # удалили признак
                    sub_features = self.feature_del(sub_features)
                    now_Q = self.calcul_Q(sub_features)
                    history_Q.append(now_Q)

                    if local_Q_best > now_Q:
                        local_Q_best = now_Q
                        local_best_sub_features = copy.deepcopy(sub_features)
                        t_0_add = t

                    if t - t_0 >= self.d_dell:
                        break

            if set(self.best_sub_features) == set(local_best_sub_features):
                return self.Q_best, self.best_sub_features

            if self.Q_best > local_Q_best:
                self.Q_best = local_Q_best
                self.best_sub_features = local_best_sub_features

            # Вернем значение наименьшей ошибки и список признаков на котором она достигается
            else:
                return self.Q_best, self.best_sub_features