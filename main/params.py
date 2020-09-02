from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.linear_model import Ridge, SGDRegressor, ElasticNet, LinearRegression
from sklearn.svm import LinearSVC, LinearSVR

random_state = 42
params = {"Classification": {"LinearSVC":
    {
        "estimator": LinearSVC(random_state=random_state),
        "params": {"penalty": ["l1", "l2"],
                   "loss": ["hinge", "squared_hinge"],
                   "C": [1.0]
                   }
    },
    "LogisticRegression":  # ?solver
        {
            "estimator": LogisticRegression(n_jobs=-1, random_state=random_state),
            "params": {"penalty": ["l1", "l2", "elasticnet", "none"],
                       "C": [1.0]
                       }
        },
    "RidgeClassifier":  # ?solver
        {
            "estimator": RidgeClassifier(random_state=random_state),
            "params": {
                "alpha": [1.0]
            }
        },
    "SGDClassifier":
        {
            "estimator": SGDClassifier(n_jobs=-1, random_state=random_state),
            "params": {"penalty": ["l1", "l2", "elasticnet"],
                       "loss": ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                       "alpha": [1.0]
                       }
        }
},
    "Regression": { 'Ridge':
    {
        'estimator': Ridge(),
        'params': {'alpha':[0.01, 0.1, 0.5, 1]}
    },

    'SGDRegressor':
        {
            'estimator': SGDRegressor(),
            'params': {'penalty' :['l2', 'l1', 'elasticnet'],
                             'alpha' : [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
                             'eta0' : [0.001, 0.01, 0.05, 0.1]
                       }
        },
    'ElasticNet':
        {
            'estimator': ElasticNet(),
            'params': {'alpha' : [0.0001, 0.001, 0.01, 0.1, 0.5, 1]}
        },
    'LinearSVR':
        {
            'estimator': LinearSVR(),
            'params': {'C' : [0.01, 0.1, 1],
                       'loss' : ['epsilon_insensitive', 'squared_epsilon_insensitive']
                       }
        },
    'LinearRegression':
        {
            'estimator': LinearRegression(),
            'params': {'n_jobs': -1}
        }
    }
}
