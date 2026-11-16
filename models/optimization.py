import numpy as np

class momentum:
	def __init__(self, beta = 0, momentum = None):
		self.beta = beta 
		self.momentum = momentum

	def new_momentum(self, gradient):
		if momentum is None:
			self.momentum = np.mean(gradient)
		self.momentum = beta * self.momentum + (1 - beta) * gradient
		return self.momentum

class RMS:
	def __init__(self, learning_rate = 0, epsilon = 1e-7):
		self.epsilon = epsilon
		self.learning_rate = learning_rate

	def new_learning_rate(self, mean_squared_gradient):
		return self.learning_rate / np.sqrt(mean_squared_gradient + self.epsilon)