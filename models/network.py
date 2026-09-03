from models import models as md
import numpy as np

class Layer:
	def __init__(self,
	activation = None
	):
		self.activation = activation
		self.bias = None
		self.weights = None
		# set last prediction for back propagation 
		self._last_prediction = None
	def feed(X):
		z = X @ self.weights + self.bias
		if not self.activation:
			self._last_prediction = z
			return z
		s = self.activation(z)
		self._last_prediction = s
		return s

class NeuralNetwork:
	def __init__(self):
		self._output_layer = None
		self._input_layer = None 
		self._hidden_layers = [] 
		
	def fit(self, X, y, hidden_layer_size = 16, output_layer_size = None):
		
		n_samples, n_features = X.shape
		self._input_layer = np.random.random((n_features, hidden_layer_size))
		
		if not output_layer_size:
			output_layer_size = y.shape[1]
		self._output_layer = np.random.random((hidden_layer_size, output_layer_size))
		sefl._initialize_hidden_layers()
		self.layers = [self._input_layer, self._hidden_layer, self._output_layer]
		
		y_prediction = self.predict(X)
		
	# recursively propagate forwards
	def predict(self, X, layer_idx = 0):
		layer_prediction = layer.feed(X)
		
		#base case
		if layer_idx < (len(self.layers) - 1):
			return layer_prediction
		return self.predict(layer_prediction, layer_idx = layer_idx + 1)
	
	"""Initialize hidden layers"""
	def _initialize_hidden_layers(self, hidden_layer_size, n_hdn_layers, activation_func = md.sigmoid):
		for _ in range(n_hdn_layers):
			self._hidden_layers.append(
				Layer(
					activation = activation_func,
					weights = np.random.random((hidden_layer_size, hidden_layer_size)),
					bias = np.random.random((hidden_layer_size,))
					)
				)