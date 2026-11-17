import numpy as np

class Momentum:
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
		return self.learning_rate / (np.sqrt(mean_squared_gradient) + self.epsilon)

class Adam:
	def __init__(self, learning_rate = 0, momentum = None):
		self.learning_rate = learning_rate
		self.moment_1 = 0
		self.moment_2 = 0
		self.beta_1 = 0.9
		self.beta_2 = 0.999
		self.momentum = momentum
		self.epsilon = 1e-7
		
	def new_weight_update(self, gradient):
		if not self.moment_1:
			self.moment_1 = np.mean(gradient)
			self.moment_2 = np.var(gradient)
			
		self.moment_1 = self.beta_1 * self.moment_1 + (1 - self.beta_2) * gradient
		self.moment_2 = self.beta_2 * self.moment_2 + (1 - self.beta_2) * gradient ** 2

def sigmoid(z, deriv = False):
	if deriv:
		s = sigmoid(z, deriv = False)
		return s * (1 - s)
	return 1 / (1 + np.exp(-z))
	
def asigmoid(z):
	#handle edgecases
	z_2 = np.select([z == 0, z == 1], [0.001, 0.999], default = z)
	return - np.log(1 / z_2 - 1)

def relu(s, deriv = False):
	if deriv:
		return (s > 0).astype(np.int32)
	return np.where(s < 0, 0, s)

def softmax(y, deriv = False):
	y_1 = np.exp(y)
	if not deriv:
		return y_1 / np.sum(y_1, axis = 0)
	s = softmax(y, deriv = False)
	return s * (1 -s)
	