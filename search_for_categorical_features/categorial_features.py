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
