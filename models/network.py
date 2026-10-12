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
		self.hidden_layer_sizes = np.array([])
		
	def fit(self, 
	X, 
	y,  
	output_layer_size = None,
	learning_rate = 0.001,
	epochs = 3000,
	batch_size = None):
		
		self.learning_rate = learning_rate
		n_samples, n_features = X.shape
		
		if not output_layer_size:
			output_layer_size = y.shape[1]
		
		self.hidden_layer_sizes = self._generate_hidden_layer_sizes(n_features, output_layer_size)
		"""Declare input, output and hidden layers"""
		# input and output
		self._initialize_io_layers(n_features, output_layer_size)
		# hidden layers
		self._initialize_hidden_layers(
			activation_func = self.activation_func,
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
	def _initialize_hidden_layers(self, activation_func = md.sigmoid):
		for i in range(len(self.hidden_layer_sizes) - 1):
			self._hidden_layers.append(
				Layer(
					activation = activation_func,
					weights = np.random.random(self.hidden_layer_sizes[i:i + 2]),
					bias = np.random.random((self.hidden_layer_sizes[i + 1],))
					)
				)
				
	"""Initialize input and output layers"""
	def _initialize_io_layers(self, n_features, output_layer_size):
		self._input_layer = Layer(
					activation = None,
					weights =  np.random.random((n_features, self.hidden_layer_sizes[0])),
					bias = np.random.random((self.hidden_layer_sizes[0],))
				)

		self._output_layer = Layer(
					activation = md.sigmoid,
					weights = np.random.random((self.hidden_layer_sizes[-1], output_layer_size)),
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
		print(f"dz_dw = {dz_dw}")
		print(f"da_dz = {da_dz}")
		print(f"delta = {delta}")
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
				loss = (a - y_batch) ** 2
				dL_da = -2 * (a - y_batch)
				print(f"a = {a}")
				print(f"dL_da = {dL_da}")
				self._backprop(delta = dL_da)
				accuracy = np.sum(self.predict(X) == y) / (y.shape[0] * y.shape[1])
				print(f"Epoch = {epoch} of {epochs}, Batch number {idx} of {len(batch_idxs) - 1}")
				print(f"Loss = {np.sum(loss):.2f}")
				print(f"Accuracy = {(accuracy * 100):.2f}%")
				
	"""Return array of sizes that are powers of 2"""
	def _generate_hidden_layer_sizes(self, n_features, output_layer_size):
		log_of_base = np.log(2)
		if n_features > output_layer_size:
			upper_limit = np.floor(np.log(n_features) / log_of_base)
			lower_limit = np.ceil(np.log(output_layer_size) / log_of_base)
			return (2 ** np.arange(upper_limit, lower_limit - 1, -1)).astype(np.int64)
		upper_limit = np.floor(np.log(output_layer_size) / log_of_base)
		lower_limit = np.ceil(np.log(n_features) / log_of_base)
		return (2 ** np.arange(lower_limit, upper_limit + 1)).astype(np.int64)
	
class CNNLayer(Layer):
	def __init__(self, k, **kwargs):
		self.k = k
		super().__init__(**kwargs)
		
	def feed(self, **kwargs):
		depth, height, width = kwargs["X"].shape
		kwargs_2 = kwargs
		kwargs_2["X"] = self._change_input_shape(kwargs["X"], depth, height, width)
		super().feed(**kwargs_2)
	
	def _change_input_shape(self, X, depth, height, width):
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
		
		return self.kernel.flatten()[:, np.newaxis]
	
class CNN(NeuralNetwork):
	def __init__(self, *kwargs):
		super().__init__(*kwargs)
		
	def fit(self, *kwargs):
		pass