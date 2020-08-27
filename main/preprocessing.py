import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def pirson(X, y=None):
    return np.corrcoef(X, y, rowvar=False)


def normalized(X):
    pass


def scale(X):
    return (X - np.mean(X)) / np.std(X)


'''
# использовать эту функцию для проверки scale с использованием numpy
# значения почти похожи, после запятой сходится несколько знаков
def scale(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
'''


def fillna(X, inds, i):
    if i == 'mean':
        col_mean = np.nanmean(X, axis=0)
        X[inds] = np.take(col_mean, inds[1])
        return X
    elif i == 'zero':
        X[inds] = 0
        return X
    else:
        raise ValueError('Error')


def cat_one_hot(X, cat_features):
    pass


def tt_split(X, y, test_size=0.3, rs=42):  # return X_train, X_test, y_train, y_test
    # избавиться от скилерновской функции, написать свою
    return train_test_split(X, y, test_size=test_size, random_state=rs)


def preprocessing(X, y, test_size=0.3, rs=42):
    # Return Pearson product-moment correlation coefficients.
    pir_matrix = pirson(X, y)

    # drop_feature_y здесь находятся индексы признаков, которые коррелируют с таргетом
    drop_feature_y = np.unique(np.where((pir_matrix[-1, :] > 0.9) & (pir_matrix[-1, :] != 1)))
    # print(drop_feature_y)

    # если drop_feature_y не пусто, значит есть признаки, которые коррелируют с таргетом - их удаляем
    if drop_feature_y.size > 0:
        X = np.delete(X, drop_feature_y, 1)
    # print(X.shape)

    X = scale(X)
    # print(X.shape)

    # np.savetxt('scale1.csv', scale1(X))
    # np.savetxt('scale2.csv', scale2(X))

    return tt_split(X, y, test_size, rs=rs)


if __name__ == '__main__':
    X = np.loadtxt('../dataset/train.csv')
    y = np.loadtxt('../dataset/train_label.csv')
    # print(X.shape, y.shape)
    preprocessing(X, y)
