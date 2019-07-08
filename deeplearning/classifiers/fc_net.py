import numpy as np

from deeplearning.layers import *
from deeplearning.layer_utils import *


class TwoLayerNet(object):
	"""
	A two-layer fully-connected neural network with ReLU nonlinearity and
	softmax loss that uses a modular layer design. We assume an input dimension
	of D, a hidden dimension of H, and perform classification over C classes.

	The architecure should be affine - relu - affine - softmax.

	Note that this class does not implement gradient descent; instead, it
	will interact with a separate Solver object that is responsible for running
	optimization.

	The learnable parameters of the model are stored in the dictionary
	self.params that maps parameter names to numpy arrays.
	"""

	def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
				 weight_scale=1e-3, reg=0.0):
		"""
		Initialize a new network.

		Inputs:
		- input_dim: An integer giving the size of the input
		- hidden_dim: An integer giving the size of the hidden layer
		- num_classes: An integer giving the number of classes to classify
		- dropout: Scalar between 0 and 1 giving dropout strength.
		- weight_scale: Scalar giving the standard deviation for random
		  initialization of the weights.
		- reg: Scalar giving L2 regularization strength.
		"""
		self.params = {}
		self.reg = reg

		############################################################################
		# Initialize the weights and biases of the two-layer net. Weights should   #
		# be initialized from a Gaussian with standard deviation equal to 		   #
		# weight_scale, and biases should be initialized to zero. All weights and  #
		# biases should be stored in the dictionary self.params, with first layer  #
		# weights and biases using the keys 'W1' and 'b1' and second layer weights #
		# and biases using the keys 'W2' and 'b2'.                                 #
		############################################################################
		self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dim])
		self.params['b1'] = np.zeros(hidden_dim)
		self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
		self.params['b2'] = np.zeros(num_classes)
		############################################################################
		#                             				                               #
		############################################################################

	def loss(self, X, y=None):
		"""
		Compute loss and gradient for a minibatch of data.

		Inputs:
		- X: Array of input data of shape (N, d_1, ..., d_k)
		- y: Array of labels, of shape (N,). y[i] gives the label for X[i].

		Returns:
		If y is None, then run a test-time forward pass of the model and return:
		- scores: Array of shape (N, C) giving classification scores, where
		  scores[i, c] is the classification score for X[i] and class c.

		If y is not None, then run a training-time forward and backward pass and
		return a tuple of:
		- loss: Scalar value giving the loss
		- grads: Dictionary with the same keys as self.params, mapping parameter
		  names to gradients of the loss with respect to those parameters.
		"""
		scores = None
		############################################################################
		# Implement the forward pass for the two-layer net, computing the class    #
		# scores for X and storing them in the scores variable.              	   #
		############################################################################
		a, hybrid_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
		scores, affine_cache = affine_forward(a, self.params['W2'], self.params['b2'])
		############################################################################
		#                             				                               #
		############################################################################

		# If y is None then we are in test mode so just return scores
		if y is None:
			return scores

		loss, grads = 0, {}
		############################################################################
		# Implement the backward pass for the two-layer net. Store the loss in the #
		# loss variable and gradients in the grads dictionary. Compute data loss   #
		# using softmax, and make sure that grads[k] holds the gradients for       #
		# self.params[k]. Then add L2 regularization. L2 regularization includes   #
		# a factor of 0.5 to simplify the expression for the gradient.             #
		############################################################################
		N = len(y)
		exp_scores = np.exp(scores)  # N*C
		softmax = np.divide(exp_scores, np.sum(exp_scores, axis=1).reshape(-1, 1))  # N*C
		t = [[j == y[i] for j in range(len(self.params['b2']))] for i in range(N)]  # N*C

		losses = -np.sum(t * np.log(softmax), axis=1)  # N
		loss = np.mean(losses) + self.reg * 0.5 * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))

		dout = softmax - t  # N*C
		da, dw2, db2 = affine_backward(dout, affine_cache)
		grads['W2'] = 1. / N * dw2 + self.reg * self.params['W2']  # M*C
		grads['b2'] = 1. / N * db2  # C
		dx, dw1, db1 = affine_relu_backward(da, hybrid_cache)
		grads['W1'] = 1. / N * dw1 + self.reg * self.params['W1']  # D*M
		grads['b1'] = 1. / N * db1  # M
		############################################################################
		#                             				                               #
		############################################################################

		return loss, grads


class FullyConnectedNet(object):
	"""
	A fully-connected neural network with an arbitrary number of hidden layers,
	ReLU nonlinearities, and a softmax loss function. This will also implement
	dropout and batch normalization as options. For a network with L layers,
	the architecture will be

	{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

	where batch normalization and dropout are optional, and the {...} block is
	repeated L - 1 times.

	Similar to the TwoLayerNet above, learnable parameters are stored in the
	self.params dictionary and will be learned using the Solver class.
	"""

	def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
				 dropout=0, use_batchnorm=False, reg=0.0,
				 weight_scale=1e-2, dtype=np.float32, seed=None):
		"""
		Initialize a new FullyConnectedNet.

		Inputs:
		- hidden_dims: A list of integers giving the size of each hidden layer.
		- input_dim: An integer giving the size of the input.
		- num_classes: An integer giving the number of classes to classify.
		- dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
		  the network should not use dropout at all.
		- use_batchnorm: Whether or not the network should use batch normalization.
		- reg: Scalar giving L2 regularization strength.
		- weight_scale: Scalar giving the standard deviation for random
		  initialization of the weights.
		- dtype: A numpy datatype object; all computations will be performed using
		  this datatype. float32 is faster but less accurate, so you should use
		  float64 for numeric gradient checking.
		- seed: If not None, then pass this random seed to the dropout layers. This
		  will make the dropout layers deteriminstic so we can gradient check the
		  model.
		"""
		self.use_batchnorm = use_batchnorm
		self.use_dropout = dropout > 0
		self.reg = reg
		self.num_layers = 1 + len(hidden_dims)
		self.dtype = dtype
		self.params = {}

		############################################################################
		# Initialize the parameters of the network, storing all values in the 	   #
		# self.params dictionary. Store weights and biases for the first layer in  #
		# W1 and b1; for the second layer use W2 and b2, etc. Weights should be    #
		# initialized from a normal distribution with standard deviation equal to  #
		# weight_scale and biases should be initialized to zero.                   #
		#                                                                          #
		# When using batch normalization, store scale and shift parameters for the #
		# first layer in gamma1 and beta1; for the second layer use gamma2 and     #
		# beta2, etc. Scale parameters should be initialized to one and shift      #
		# parameters should be initialized to zero.                                #
		############################################################################
		if type(hidden_dims) != list:
			raise ValueError('hidden_dim has to be a list.')

		dims = [input_dim] + hidden_dims + [num_classes]
		for i in range(self.num_layers):
			self.params['W' + str(i + 1)] = np.random.normal(0, weight_scale, [dims[i], dims[i + 1]])
			self.params['b' + str(i + 1)] = np.zeros(dims[i + 1])

			if self.use_batchnorm and i != (self.num_layers - 1):
				self.params['gamma' + str(i + 1)] = np.ones(dims[i + 1])
				self.params['beta' + str(i + 1)] = np.zeros(dims[i + 1])
		############################################################################
		#                             				                               #
		############################################################################

		# When using dropout we need to pass a dropout_param dictionary to each
		# dropout layer so that the layer knows the dropout probability and the mode
		# (train / test). We can pass the same dropout_param to each dropout layer.
		self.dropout_param = {}
		if self.use_dropout:
			self.dropout_param = {'mode': 'train', 'p': dropout}
			if seed is not None:
				self.dropout_param['seed'] = seed

		# With batch normalization we need to keep track of running means and
		# variances, so we need to pass a special bn_param object to each batch
		# normalization layer. We pass self.bn_params[0] to the forward pass of
		# the first batch normalization layer, self.bn_params[1] to the forward
		# pass of the second batch normalization layer, etc.
		self.bn_params = []
		if self.use_batchnorm:
			self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

		# Cast all parameters to the correct datatype
		for k, v in self.params.items():
			self.params[k] = v.astype(dtype)

	def loss(self, X, y=None):
		"""
		Compute loss and gradient for the fully-connected net.

		Input / output: Same as TwoLayerNet above.
		"""
		X = X.astype(self.dtype)
		mode = 'test' if y is None else 'train'

		# Set train/test mode for batchnorm params and dropout param since they
		# behave differently during training and testing.
		if self.dropout_param is not None:
			self.dropout_param['mode'] = mode
		if self.use_batchnorm:
			for bn_param in self.bn_params:
				bn_param[mode] = mode

		scores = None
		############################################################################
		# Implement the forward pass for the fully-connected net, computing the    #
		# class scores for X and storing them in the scores variable.          	   #
		#                                                                          #
		# When using dropout, we'll need to pass self.dropout_param to each        #
		# dropout forward pass.                                                    #
		#                                                                          #
		# When using batch normalization, you'll need to pass self.bn_params[0] to #
		# the forward pass for the first batch normalization layer, pass           #
		# self.bn_params[1] to the forward pass for the second batch normalization #
		# layer, etc.                                                              #
		############################################################################
		X_hid = [None] * (self.num_layers - 1)
		affine_caches = [None] * self.num_layers
		batchnorm_caches = [None] * (self.num_layers - 1)
		relu_caches = [None] * (self.num_layers - 1)
		dropout_caches = [None] * (self.num_layers - 1)

		a, affine_caches[0] = affine_forward(X, self.params['W1'], self.params['b1'])
		if self.use_batchnorm:
			a, batchnorm_caches[0] = batchnorm_forward(a, self.params['gamma1'], self.params['beta1'],
													   self.bn_params[0])
		X_hid[0], relu_caches[0] = relu_forward(a)
		if self.use_dropout:
			X_hid[0], dropout_caches[0] = dropout_forward(X_hid[0], self.dropout_param)
		for i in range(self.num_layers - 2):
			a, affine_caches[i + 1] = affine_forward(X_hid[i], self.params['W' + str(i + 2)],
													 self.params['b' + str(i + 2)])
			if self.use_batchnorm:
				a, batchnorm_caches[i + 1] = batchnorm_forward(a, self.params['gamma' + str(i + 2)],
															   self.params['beta' + str(i + 2)], self.bn_params[i + 1])
			X_hid[i + 1], relu_caches[i + 1] = relu_forward(a)
			if self.use_dropout:
				X_hid[i + 1], dropout_caches[i + 1] = dropout_forward(X_hid[i + 1], self.dropout_param)
		scores, affine_caches[-1] = affine_forward(X_hid[-1], self.params['W' + str(self.num_layers)],
												   self.params['b' + str(self.num_layers)])
		############################################################################
		#                             				                               #
		############################################################################

		# If test mode return early
		if mode == 'test':
			return scores

		loss, grads = 0.0, {}
		############################################################################
		# Implement the backward pass for the fully-connected net. Store the loss  #
		# in the loss variable and gradients in the grads dictionary. Compute data #
		# loss using softmax, and make sure that grads[k] holds the gradients for  #
		# self.params[k]. Then add L2 regularization. 			   				   #
		# L2 regularization includes a factor of 0.5 to simplify the expression    #
		# for the gradient.   													   #
		#                                                                          #
		# When using batch normalization, we don't need to regularize the scale    #
		# and shift parameters.                                                    #
		############################################################################
		loss, dout = softmax_loss(scores, y)
		loss += self.reg * 0.5 * sum([np.sum(self.params['W' + str(i + 1)] ** 2) for i in range(self.num_layers)])

		da, dw, db = affine_backward(dout, affine_caches[-1])
		grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]  # M*C
		grads['b' + str(self.num_layers)] = db  # C
		if self.use_dropout:
			da = dropout_backward(da, dropout_caches[-1])
		dx = relu_backward(da, relu_caches[-1])
		if self.use_batchnorm:
			dx, grads['gamma' + str(self.num_layers - 1)], grads[
				'beta' + str(self.num_layers - 1)] = batchnorm_backward_alt(dx, batchnorm_caches[-1])
		for i in range(self.num_layers - 2):
			da, dw, db = affine_backward(dx, affine_caches[-(i + 2)])
			grads['W' + str(self.num_layers - 1 - i)] = dw + self.reg * self.params[
				'W' + str(self.num_layers - 1 - i)]  # M*C
			grads['b' + str(self.num_layers - 1 - i)] = db  # C
			if self.use_dropout:
				da = dropout_backward(da, dropout_caches[-(i + 2)])
			dx = relu_backward(da, relu_caches[-(i + 2)])
			if self.use_batchnorm:
				dx, grads['gamma' + str(self.num_layers - 2 - i)], grads[
					'beta' + str(self.num_layers - 2 - i)] = batchnorm_backward_alt(dx, batchnorm_caches[-(i + 2)])
		_, dw, db = affine_backward(dx, affine_caches[0])
		grads['W1'] = dw + self.reg * self.params['W1']  # M*C
		grads['b1'] = db  # C
		############################################################################
		#                             				                               #
		############################################################################

		return loss, grads

	def save(self, path=".\\model.txt"):
		import json
		with open(path, 'w') as file:
			file.write(json.dumps(self.params))

	def load(self, path=".\\model.txt"):
		import json
		self.params = json.load(open(path))

