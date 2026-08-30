from models import models as md
import numpy as np

class NeuralNetwork:
	def __init__(self):
		self.hidden_layers = None
		self.n_hidden_layers = None
		self.output_layer = None 
		self.input_layer = None
		
	def _feed_forward(self, X):
		pass
	
	def _feed_backward(self):
		pass
	
	def fit(self, X, y):
		n_samples, n_features = X.shape
		self.input_layer = np.zeros(n_features)
		pass
	
	def predict(self, X):
		pass