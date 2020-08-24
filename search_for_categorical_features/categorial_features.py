import numpy as np

def seach_cat_features(data, k=10):

    '''
    k - порог принадлежности к категориальным признакам
    по числу уникальных значений в признаке.
    '''
    cat_list = []
    for i in range(data.shape[1]):
        if type(data[1, i]) == np.str_:
            cat_list.append(i)
        if np.unique(data[:, i]).shape[0] <= k:
            cat_list.append(i)
    return cat_list


def one_hot_encoder(data, num_feature):
    # создаем массив из нуляей
    A = np.zeros((data.shape[0], np.unique(data[:, num_feature]).shape[0]))
    # создаем словарь {Значение кат.признака : число}
    dict_for_dummy = {}
    dict_for_dummy = {j: i for i, j in enumerate(np.unique(data[:, num_feature]))}
    for i in range(data.shape[0]):
        A[i, dict_for_dummy[data[i, num_feature]]] = 1

    return A

