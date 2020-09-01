from sklearn.linear_model import RidgeClassifier, LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

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
    "Regression": {}
}
