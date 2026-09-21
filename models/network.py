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
		"""Cache for backprop"""
		self.a = None # a = activation(z)
		self.z = None # z = Xw + b
		self.a_1 = None
	def feed(self, a_1):
		self.a_1 = a_1 # for backprop
		self.z = a_1 @ self.weights + self.bias
		if not self.activation:
			self.a = self.z
			return self.z
		self.a = self.activation(self.z)
		return self.a

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
	learning_rate = 0.001,
	epochs = 3000,
	batch_size = None):
		
		self.learning_rate = learning_rate
		n_samples, n_features = X.shape
		
		if not output_layer_size:
			output_layer_size = y.shape[1]
		
		"""Declare input, output and hidden layers"""
		# input and output
		self._initialize_layers(n_features, hidden_layer_size, output_layer_size)
		# hidden layers
		self._initialize_hidden_layers(
			activation_func = self.activation_func,
			hidden_layer_size = hidden_layer_size,
			n_hdn_layers = n_hdn_layers
		)

		self.layers = [self._input_layer, *self._hidden_layers, self._output_layer]
		self._mini_batch(X, y, epochs, batch_size)
		
	"Recursively feed forward"
	def _feed_forward(self, X, layer_idx = 0):
		layer = self.layers[layer_idx]
		layer_prediction = layer.feed(X)
		
		#base case
		if layer_idx > (len(self.layers) - 2):
			return layer_prediction
			
		return self._feed_forward(layer_prediction, layer_idx = layer_idx + 1)
	
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
					activation = md.sigmoid,
					weights = np.random.random((hidden_layer_size, output_layer_size)),
					bias = np.random.random((output_layer_size,))
					)
	
	def _backprop(self, delta, layer_idx = None):
		layer_idx = len(self.layers) - 1 if (layer_idx is None) else layer_idx
		# base case 
		if layer_idx < 0:
			return
		layer = self.layers[layer_idx]
		
		#gradients
		da_dz = layer.activation(layer.z, deriv = True) if layer.activation else 1
		dz_dw = layer.a_1.T
		
		local_delta = da_dz * delta
		
		dL_dw = dz_dw @ local_delta
		da_da1 = local_delta @ layer.weights.T
		# update 
		n = 1  / layer.a_1.shape[0]
		layer.weights -= n * self.learning_rate * dL_dw
		layer.bias -= n * self.learning_rate * np.sum(local_delta, axis = 0)
		
		self._backprop(delta = da_da1, layer_idx = layer_idx - 1)
		
	def predict(self, X):
		return np.round(self._feed_forward(X))
		
	def _mini_batch(self, X, y, epochs, batch_size = 128):
		n_samples, _ = X.shape
		batch_idxs = [*range(0, n_samples, batch_size), n_samples]
		
		
		for epoch in range(epochs):
			for idx in range(len(batch_idxs) - 1):
				X_batch = X[batch_idxs[idx]:batch_idxs[idx + 1]]
				y_batch = y[batch_idxs[idx]:batch_idxs[idx + 1]]
				a = self._feed_forward(X_batch)
				loss = (y_batch - a) ** 2
				dL_da = 2 * (a - y_batch)
				self._backprop(delta = dL_da)
			if epoch % 20 == 0:
				accuracy = np.sum(self.predict(X) == y) / (y.shape[0] * y.shape[1])
				print(f"Loss = {np.sum(loss):.2f}")
				print(f"Accuracy = {(accuracy * 100):.2f}%")
				
		
class CNN:
	def __init__(self,
	k = None,
	):
		self.kernel = None 
		self.X = np.array([]) # for for loss function 
		self.k = k # kernel height or width 
		self.bias = None 
		
	def fit(self, X, y):
		height, width = X.shape
		
		if self.k is None:
			self.k = height // 2
			
		self.kernel = np.random.random((self.k, self.k))
		
	def cross_correlation(self, X, valid = True):
		height, width = X.shape
		
		if valid: # valid cross_correlation
			image = X
		else: # full cross_correlation
			image = np.zeros((height + 2 * self.k - 2, width + 2 * self.k - 2))
			image[self.k - 1: self.k - 1 + height, self.k - 1:self.k -1 + width] = X
			height, width = image.shape
			
		
		for i in range(height - self.k + 1):
			for j in range(width - self.k + 1):
				a = image[i:i + self.k, j:j + self.k].flatten()
				self.X = np.append(self.X, a)
		self.X = self.X.reshape((-1, self.k ** 2))
		
		return self.X @ self.kernel.flatten()[:, np.newaxis] # + self.bias