from models import models as md
import numpy as np

class NeuralNetwork:
	def __init__(self):
		self._output_layer = None
		self._input_layer = None 
		self._hidden_layer = None 
		
	def fit(self, X, y, hidden_layer_size = 16, output_layer_size = None):
		
		n_samples, n_features = X.shape
		self._input_layer = np.random.random((n_features, hidden_layer_size))
		self._hidden_layer = np.random.random((hidden_layer_size, hidden_layer_size))
		
		if not output_layer_size:
			output_layer_size = y.shape[1]
		self._output_layer = np.random.random((hidden_layer_size, output_layer_size))
		
		self.layers = [self._input_layer, self._hidden_layer, self._output_layer]
		
		y_prediction = self.predict(X)
		
	# recursively propagate forwards
	def predict(self, X, layer_idx = 0):
		layer_prediction = md.sigmoid(X @ self.layers[layer_idx])
		#base case
		if layer_idx < (len(self.layers) - 1):
			return layer_prediction
		return self.predict(layer_prediction, layer_idx = layer_idx + 1)