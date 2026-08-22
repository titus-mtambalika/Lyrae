import numpy as np
import math

def track_loss(model,
X, 
y, 
epochs = 5000,
i = 50, 
learning_rate = 0.01,
gradient = False,
relative = False,
prediction_function = None):
	if prediction_function is None:
		prediction_function = model.predict
	loss_report = np.array([])
	# epochs for total iterations i for epochs per iterations
	for _ in range(epochs // i):
		model.fit(X, y, epochs = i, learning_rate = learning_rate)
		loss = np.mean((y - prediction_function(X)) ** 2)
		loss_report = np.append(loss_report, loss)
	# to track approximate gradient of loss curve
	if gradient:
		loss_report = loss_report[1:] - loss_report[:-1]
	# make initial loss 1 to compare gradient change
	if relative:
		loss_report = loss_report / loss_report[0]
	return loss_report, np.linspace(0, epochs, len(loss_report))

def normalize(X, axis = 0):
	return (X - np.mean(X, axis = axis)) / np.std(X, axis = axis)

class Lin_reg:
	def __init__(self):
		self.weights = None
		self.bias = None
	
	def fit(self, X, y, epochs = 3000, learning_rate = 0.01):
		n_samples, n_features = X.shape
		self.loss_report = np.array([])
		if self.weights is None:
			self.weights = np.zeros((n_features, 1))
		if self.bias is None:
			self.bias = 0 
			
		for epoch in range(epochs):
			prediction = X @ self.weights + self.bias
			
			self.weights -= (learning_rate / n_samples) * (X.T @ (prediction - y))
			self.bias -= learning_rate * np.mean(prediction - y)
			
	def predict(self, X):
		return X @ self.weights + self.bias

class Log_reg:
	def __init__(self):
		self.weights = None
		self.bias = None
		
	def fit(self, X, y, epochs = 50000, learning_rate=0.01):
		n_samples, n_features = X.shape
		self.weights = np.zeros((n_features, 1))
		self.bias = 0
		
		for epoch in range(epochs):
			z = X @ self.weights + self.bias
			y_prediction = self._sigmoid(z)
			
			# Gradients (simplified form)
			dl_dweights = X.T @ (y_prediction - y)
			dl_dbias = np.sum(y_prediction - y)
			
			# Update parameters
			self.weights -= learning_rate * dl_dweights
			self.bias -= learning_rate * dl_dbias
	
	"""Return prediction as probabilities"""
	def predict_proba(self, X):
		z = X @ self.weights + self.bias
		return self._sigmoid(z) 
	
	"""Return prediction as 0 or 1"""
	def predict(self, X, threshold=0.5):
		proba = self.predict_proba(X)
		return (proba >= threshold).astype(int)

class KNN:
	def __init__(self, X, y):
		self.X = X 
		self.y = y
		
	"""Return class with highest likelyhood of sample membership"""
	def _classify_one(self, sample, K = 7):
		distances = np.sum((sample - self.X) ** 2, axis = 1)
		nearest_neighbours = self.y[np.argsort(distances)]
		labels, votes = np.unique(nearest_neighbours[:K], return_counts = True)
		return labels[np.argmax(votes)]
		
	def classify(self, samples, K = 7):
		return [self._classify_one(sample) for sample in samples]
		
	def predict_one(self, sample, K = 7):
		distances = np.sum((sample - self.X) ** 2, axis = 1)
		nearest_neighbours = self.y[np.argsort(distances)]
		votes = nearest_neighbours[:K]
		return np.mean(votes)
		
	def predict(self, samples, K = 7):
		return [self.predict_one(sample) for sample in samples]
		
	def sigmoid(self, z):
		return 1 / (1 + math.e ** -z)

class GaussianNB:
	def __init__(self, epsilon=1e-9):
		self.epsilon = epsilon
		self.classes = None
		self.priors = None
		self.means = None
		self.variances = None
		self._log_priors = None  # cache
	
	def fit(self, X, y):
		# Ensure y is 1D
		y = y.flatten() if y.ndim > 1 else y
		
		self.classes = np.unique(y)
		self.priors = {}
		self.means = {}
		self.variances = {}
		
		for c in self.classes:
			X_c = X[y == c]
			self.priors[c] = len(X_c) / len(X)
			self.means[c] = np.mean(X_c, axis=0)
			self.variances[c] = np.var(X_c, axis=0) + self.epsilon
		
		# Cache for faster predictions
		self._log_priors = {c: np.log(self.priors[c]) for c in self.classes}
		self._log_variances = {c: np.log(self.variances[c]) for c in self.classes}
		self._d = X.shape[1]  # number of features
	
	def _log_likelihood_batch(self, X, c):
		"""Returns log likelihood for all samples in X for class c"""
		# Shape: (m,)
		log_density = -0.5 * self._d * np.log(2 * np.pi) - \
					 0.5 * np.sum(self._log_variances[c]) - \
					 0.5 * np.sum((X - self.means[c])**2 / self.variances[c], axis=1)
		return self._log_priors[c] + log_density
	
	def predict_proba(self, X):
		"""Returns class probabilities for all samples"""
		# Shape: (m, C)
		scores = np.column_stack([
			self._log_likelihood_batch(X, c) 
			for c in self.classes
		])
		
		# Numerically stable softmax
		scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
		return np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=1, keepdims=True)
	
	def predict(self, X):
		"""Returns class predictions"""
		proba = self.predict_proba(X)
		return self.classes[np.argmax(proba, axis=1)]

class GaussianNB_predictor(GaussianNB):
	def fit(self, X, y, n_classes = 10):
		# initalize upper and lower bounds for classes
		self.class_bounds = np.linspace(y[np.argmin(y)], y[np.argmax(y)], n_classes + 1)
		# classes are averages of upper and lower class_bounds
		self.classes = (self.class_bounds[:-1] + self.class_bounds[1:]) * 0.5
		super().fit(X, self._convert_to_classes(y))
		
	"""Return weighted sum of log_density"""
	def predict_batch(X):
		preictions = np.zeros(X.shape[0])
		for j, sample in enumerate(X):
			sample_prediction = np.zeros(len(self.classes))
			for i, c in enumerate(self.classes):
				sample_prediction[i] = self._log_likelihood_batch(sample, c)
			preictions[j] = sample_prediction[np.newaxis, :] @ self.classes[:, np.newaxis] / np.sum(sample_prediction)
		return preictions
		
	def _convert_to_classes(self, y):
		# handle edgecases 
		y_to_class = np.select([y < self.class_bounds[0], y > self.class_bounds[-1]], [self.classes[0], self.classes[-1]], default = y)
		conditions = [(y_to_class > i) & (y_to_class < j) for i, j in zip(self.class_bounds[:-1], self.class_bounds[1:])]
		y_to_class = np.select(conditions, self.classes, default = y_to_class)
		return y_to_class

class Node:
	def __init__(self, 
	feature = None, 
	threshold = None, 
	value = None,
	left = None,
	right = None):
		self.feature = feature
		self.threshold = threshold
		self.value = value
		self.left = left
		self.right = right
		
	@property
	def is_leaf_node(self):
		return self.value is not None
		
	def __str__(self):
		return self._generate_tree_string()
	
	"""Recursively print tree structure"""
	def _generate_tree_string(self, depth = 0, increment = 2):
		if not self.is_leaf_node:
			return f"""|-{"-" * depth}{self.feature} > {self.threshold}?
|
|
|--{self.left._generate_tree_string(depth + increment)}
|
|--{self.right._generate_tree_string(depth + increment)}"""
		else:
			return f"{"-" * depth}{self.threshold} ? --> {self.value}"

class DecisionTreeClassifier:
	def __init__(self,
	n_features = None,
	max_depth = 100,
	min_sample_split = 2):
		self.n_features = n_features
		self.max_depth = max_depth
		self.min_sample_split = min_sample_split
		self.root = None
		
	def fit(self, X, y):
		_, features = X.shape
		self.n_features = features if not self.n_features else min (features, self.n_features6)
		self.root = self._grow_tree(X, y)
	
	def predict(self, x):
		current_node = self.tree
		# traverse tree
		while not current_node.is_leaf_node:
			current_node = current_node.right if x > current_node.threshold else current_node.left
		else:
			return current_node.value
	
	def _gini_impurity(self, y):
		_, counts = np.unique(y, return_counts = True)
		return 1 - np.sum((counts / len(y)) ** 2)
		
	def _grow_tree(self, X, y, depth = 0):
		n_samples, n_features = X.shape
		n_labels = len(np.unique(y))
		
		# check stopping criteria
		if (depth >= self.max_depth) or \
		(n_features <= self.n_features) or \
		(n_samples < self.min_sample_split):
			labels, counts = np.unique(y, return_counts = True)
			return Node(value = labels[np.argmax(counts)])
			
		# find best split 
		best_threshold, best_feature_i = self.best_split(X, y)
		left_mask = X[:, best_feature_i] > best_threshold
		right_mask = ~left_mask
		
		# create child nodes
		node = Node(threshold = best_threshold, feature = best_feature_i)
		node.left = Node(self._grow_tree(X[left_mask], y[left_mask], depth = depth + 1))
		node.right = Node(self._grow_tree(X[right_mask], y[right_mask], depth = depth + 1))
		
		return node
		
	def _best_split(self, X, y):
		best_feature_i = None
		best_threshold = None
		least_impurity = 1
		
		# loop through each feature
		for feature_i in range(self.n_features):
			features = np.unique(X[:, feature_i])
			thresholds = np.argsort(features)
			# use midpoints as thresholds
			thresholds = (thresholds[:-1] + thresholds[1:]) * 0.5 
			
			# loop through each threshold
			for threshold in thresholds:
				current_gini = self._gini_impurity(y[features > threshold])
				if current_gini < least_impurity:
					least_impurity, best_threshold, best_feature_i = current_gini, threshold, feature
	
			return best_threshold, best_feature_i
			
	def __str__(self):
		return str(self.root)