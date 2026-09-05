from models import models as md
import numpy as np

class Layer:
	def __init__(self, 
	activation = None,
	weights = None,
	bias = None):
		self.activation = activation
		self.bias = bias
		self.weights = weights
		# set last prediction for back propagation 
		self._last_prediction = None
	def feed(self, X):
		z = X @ self.weights + self.bias
		if not self.activation:
			self._last_prediction = z
			return z
		s = self.activation(z)
		self._last_prediction = s
		return s

class NeuralNetwork:
	def __init__(self,
	activation_func = md.sigmoid # for hidden layers
	):
		self._output_layer = None
		self._input_layer = None 
		self._hidden_layers = []
		self.learning_rate = None
		self.activation_func = activation_func
		
	def fit(self, 
	X, 
	y, 
	hidden_layer_size = 16, 
	n_hdn_layers = 1,
	output_layer_size = None,
	learning_rate = 0.001):
		
		n_samples, n_features = X.shape
		
		if not output_layer_size:
			output_layer_size = y.shape[1]
		
		"""Declare input, output and hidden layers"""
		# input and output
		self._initialize_layers(n_features, hidden_layer_size, output_layer_size)
		# hidden layere
		self._initialize_hidden_layers(
			activation_func = self.activation_func,
			hidden_layer_size = hidden_layer_size,
			n_hdn_layers = n_hdn_layers
		)
		
		self.layers = [self._input_layer, *self._hidden_layers, self._output_layer]
		
		y_prediction = self.predict(X)
		
		print(y_prediction.shape)
	"Recursively feed forward"
	def predict(self, X, layer_idx = 0):
		layer = self.layers[layer_idx]
		layer_prediction = layer.feed(X)
		
		#base case
		if layer_idx > (len(self.layers) - 2):
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
			
	def _initialize_layers(self, n_features, hidden_layer_size, output_layer_size):
		self._input_layer = Layer(
					activation = None,
					weights =  np.random.random((n_features, hidden_layer_size)),
					bias = np.random.random((hidden_layer_size,))
				)

		self._output_layer = Layer(
					activation = md.softmax,
					weights = np.random.random((hidden_layer_size, output_layer_size)),
					bias = np.random.random((output_layer_size,))
					)