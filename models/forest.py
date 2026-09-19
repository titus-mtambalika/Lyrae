import numpy as np
from models import models as md

class RandomForest:
	def __init__(self, 
	n_trees = 6, 
	max_depth = 10, 
	min_sample_split = 7, 
	n_features = None):
		self.n_trees = n_trees
		self.max_depth = max_depth
		self.min_sample_split = min_sample_split
		self.n_features = n_features
		self.trees = []
	
	def fit(self, X, y, batch_size = None):
		n_samples, n_features = X.shape
		batch_size = n_samples // self.n_trees if not batch_size else batch_size
		self.n_features = n_features if not self.n_features else self.n_features
		
		# initialize trees 
		self.trees = [DecisionTreeClassifier(n_features = self.n_features, max_depth = self.max_depth, min_sample_split = self.min_sample_split) for _ in range(self.n_trees)]
		"""Fit each tree"""
		for tree in self.trees:
			# select random datasets
			random_indices = np.random.choice(n_samples, batch_size, replace = False)
			tree.fit(X[random_indices], y[random_indices])
		
	def predict(self, X):
		predictions = np.array([])
		# loop through each sample
		for x in X:
			sample_prediction = np.array([tree.predict(x) for tree in self.trees])
			labels, counts = np.unique(sample_prediction, return_counts = True)
			predictions = np.append(predictions, labels[np.argmax(counts)])
		return predictions

	def __str__(self):
		return "\n".join([str(tree) for tree in self.trees])