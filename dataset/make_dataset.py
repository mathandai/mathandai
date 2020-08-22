import numpy as np
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=12000, 
						   n_features=2, 
						   n_informative=2, 
						   n_redundant=0, 
						   n_repeated=0, 
						   n_classes=2, 
						   n_clusters_per_class=2, 
						   weights=None, 
						   flip_y=0.01, 
						   class_sep=1.0, 
						   hypercube=True, 
						   shift=0.0, 
						   scale=1.0, 
						   shuffle=True, 
						   random_state=42)				   
np.savetxt('train.csv', X)
np.savetxt('train_label.csv', y, fmt='%i')