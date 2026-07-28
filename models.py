import numpy as np
import math
from collections import Counter

def track_loss(model,
               X, 
               y, 
               epochs = 1000,
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
		loss = np.sum((y - prediction_function(X)) ** 2) / y.shape[0]
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

def sigmoid(z):
	return 1 / (1 + math.e ** -z)
	
def asigmoid(z):
	return -1 * np.log((1 / z) - 1)

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
			y_prediction = sigmoid(z)
			
			# Gradients (simplified form)
			dl_dweights = X.T @ (y_prediction - y)
			dl_dbias = np.sum(y_prediction - y)
			
			# Update parameters
			self.weights -= learning_rate * dl_dweights
			self.bias -= learning_rate * dl_dbias
	
	def predict_proba(self, X):
		z = X @ self.weights + self.bias
		return sigmoid(z) 
	
	def predict(self, X, threshold=0.5):
		proba = self.predict_proba(X)
		return (proba >= threshold).astype(int)

class KNN:
	def __init__(self, X, y):
		self.X = X 
		self.y = y
		
	def classify_one(self, sample, K = 7):
		distances = np.sum((sample - self.X) ** 2, axis = 1)
		nearest_neighbours = self.y[np.argsort(distances)]
		votes = nearest_neighbours[:K]
		return Counter(votes).most_common()[0][0]
		
	def classify(self, samples, K = 7):
		return [self.classify_one(sample) for sample in samples]
		
	
	def predict_one(self, sample, K = 7):
		distances = np.sum((sample - self.X) ** 2, axis = 1)
		nearest_neighbours = self.y[np.argsort(distances)]
		votes = nearest_neighbours[:K]
		return np.mean(votes)
		
	def predict(self, samples, K = 7):
		return [self.predict_one(sample) for sample in samples]

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
	def fit(self, X, y):
		self.generate_classes(y)
		super.fit(X, self.convert_to_classes(y))
		
	def generate_classes(self, y, n_classes = 10):
		# define lower and upper limits for each class
		self.classes = np.linspace(y[np.argmin(y)], y[np.argmax(y)], n_classes + 1)
	
	'''Return weighted sum of log_density'''
	def predict(X):
		sample_prediction = np.zeros(len(self.shape))
		for i, c in enumerate(self.classes):
			sample_prediction[i] = self._log_likelihood_batch(X, c)
		return sample_prediction[np.newaxis, :] @ self.classes[:, np.newaxis] / np.sum(sample_prediction)
		
	def convert_to_classes(y):
		y_to_class = y.copy()
		for i in range(self.classes.shape - 1):
			y_to_class = np.where((y_to_class < self.classes[i]) & (y_to_class < self.classes[i + 1]), self.classes[i] / 2 + self.classes[i + 1], y_to_class)
		return y_to_class

class DecisionTree:
	def __init__(self, max_depth = 10, min_sample_split = 2):
		self.max_depth = max_depth
		self.min_sample_split = min_sample_split
		self.tree = None
	
	def fit(X, y):
		pass