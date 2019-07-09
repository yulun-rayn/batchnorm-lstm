import numpy as np


""" basics """
def relu(x):
  return np.maximum(x, 0)


def tanh(x):
  return np.tanh(x)


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


""" layers """
def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # Implement the affine forward pass. Store the result in out.               #
    # Will need to reshape the input into rows.                                 #
    #############################################################################
    z = x.reshape(np.shape(x)[0], -1)  # N*D
    out = np.dot(z, w) + b  # N*M
    #############################################################################
    #                                                                           #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    #############################################################################
    # Implement the affine backward pass.                                       #
    #############################################################################
    shapes = np.shape(x)
    N = shapes[0]
    z = x.reshape(N, -1)

    dx = np.dot(dout, w.T).reshape(shapes)
    dw = np.dot(z.T, dout)
    db = np.dot(np.ones(N), dout)
    # Note: Gradient here is the sum of the gradients of N data
    #############################################################################
    #                                                                           #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    #############################################################################
    # Implement the ReLU forward pass.                                          #
    #############################################################################
    out = relu(x)
    #############################################################################
    #                                                                           #
    #############################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    #############################################################################
    # Implement the ReLU backward pass.                                         #
    #############################################################################
    dx = np.greater(x, 0) * dout
    #############################################################################
    #                                                                           #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # Implement the training-time forward pass for batch normalization.         #
        #                                                                           #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # Store the output in the variable out.                                     #
        # Any intermediates that are need for the backward pass should be stored    #
        # in the cache variable.                                                    #
        #                                                                           #
        # Use computed sample mean and variance together with the momentum          #
        # variable to update the running mean and running variance, then            #
        # store the result in the running_mean and running_var variables.           #
        #############################################################################
        sample_mean = np.mean(x, axis=0)  # D
        sample_var = np.var(x, axis=0)  # D
        x_norm = (x - sample_mean) / np.sqrt(sample_var + eps)  # N*D
        out = gamma * x_norm + beta  # N*D
        cache = (x_norm, gamma, beta, sample_mean, sample_var, x, eps)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #############################################################################
        #                                                                           #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # Implement the test-time forward pass for batch normalization.             #
        # Use the running mean and variance to normalize the incoming data,         #
        # then scale and shift the normalized data using gamma and beta.            #
        # Store the result in the out variable.                                     #
        #############################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        #############################################################################
        #                                                                           #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # Implement the backward pass for batch normalization.                      #
    # Store the results in the dx, dgamma, and dbeta variables.                 #
    #############################################################################
    x_norm, gamma, beta, sample_mean, sample_var, x, eps = cache
    x_mu = x - sample_mean
    std_mod = (sample_var + eps) ** -0.5
    N, _ = x.shape

    dx_norm = dout * gamma # N*D
    dsample_var = -0.5 * np.sum(dx_norm * x_mu, axis=0) * std_mod ** 3 # D
    dsample_mean = -1. * np.sum(dx_norm * std_mod, axis=0) - 2./N * dsample_var * np.sum(x_mu, axis=0) # D
    dx = dx_norm * std_mod + 2./N * dsample_var * x_mu + 1./N * dsample_mean
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    #############################################################################
    #                                                                           #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # Implement the backward pass for batch normalization.                      #
    # Store the results in the dx, dgamma, and dbeta variables.                 #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, we      #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement.                                                         #
    #############################################################################
    x_norm, gamma, beta, sample_mean, sample_var, x, eps = cache
    x_mu = x - sample_mean
    std_mod = (sample_var + eps) ** -0.5
    N, _ = x.shape

    dx_norm = dout * gamma  # N*D
    dx = std_mod * (dx_norm - 1./N * (x_mu * np.sum(dx_norm * x_mu * std_mod ** 2, axis=0) + np.sum(dx_norm, axis=0)))
    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)
    #############################################################################
    #                                                                           #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
      np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # Implement the training phase forward pass for inverted dropout.         #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
        ###########################################################################
        #                                                                         #
        ###########################################################################
    elif mode == 'test':
        ###########################################################################
        # Implement the test phase forward pass for inverted dropout.             #
        ###########################################################################
        out = x
        ###########################################################################
        #                                                                         #
        ###########################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # Implement the training phase backward pass for inverted dropout.        #
        ###########################################################################
        dx = dout * mask
        ###########################################################################
        #                                                                         #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # Implement the convolutional forward pass.                                 #
    #############################################################################
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    s = stride
    out = np.zeros((N, F, H_new, W_new))
    for i in range(N):       # ith image
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]) + b[f]
    #############################################################################
    #                                                                           #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # Implement the convolutional backward pass.                                #
    #############################################################################
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + (H + 2 * pad - HH) / stride
    W_new = 1 + (W + 2 * pad - WW) / stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in range(N):       # ith image
        for f in range(F):   # fth filter
            for j in range(H_new):
                for k in range(W_new):
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]
                    dw[f] += window * dout[i, f, j, k]
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]

    # Unpad
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]
    #############################################################################
    #                                                                           #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # Implement the max pooling forward pass                                    #
    #############################################################################
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = int(1 + (H - HH) / s)
    W_new = int(1 + (W - WW) / s)
    out = np.zeros((N, C, H_new, W_new))
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)
    #############################################################################
    #                                                                           #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # Implement the max pooling backward pass                                   #
    #############################################################################
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = int(1 + (H - HH) / s)
    W_new = int(1 + (W - WW) / s)
    dx = np.zeros_like(x)
    for i in range(N):
        for j in range(C):
            for k in range(H_new):
                for l in range(W_new):
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]
                    m = np.max(window)
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == m) * dout[i, j, k, l]#only max x has dx,else = 0
    #############################################################################
    #                                                                           #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # Implement the forward pass for spatial batch normalization.               #
    #############################################################################
    N, C, H, W = x.shape
    x_new = x.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    out, cache = batchnorm_forward(x_new, gamma, beta, bn_param)
    out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    #############################################################################
    #                                                                           #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # Implement the backward pass for spatial batch normalization.              #
    #############################################################################
    N, C, H, W = dout.shape
    dout_new = dout.transpose(0, 2, 3, 1).reshape(N*H*W, C)
    dx, dgamma, dbeta = batchnorm_backward(dout_new, cache)
    dx = dx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    #############################################################################
    #                                                                           #
    #############################################################################

    return dx, dgamma, dbeta


# new layers
def tanh_forward(x):
    """
    Computes the forward pass for a layer of tanh units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: tanh(x)
    """
    out = None
    #############################################################################
    # Implement the tanh forward pass.                                          #
    #############################################################################
    out = tanh(x)
    #############################################################################
    #                                                                           #
    #############################################################################
    cache = out
    return out, cache


def tanh_backward(dout, cache):
    """
    Computes the backward pass for a layer of tanh units.

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Values from the forward pass, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, tanh_x = None, cache
    #############################################################################
    # Implement the tanh backward pass.                                         #
    #############################################################################
    dx = (1 - tanh_x ** 2) * dout
    #############################################################################
    #                                                                           #
    #############################################################################
    return dx


def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoid units.

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: sigmoid(x)
    """
    out = None
    #############################################################################
    # Implement the Sigmoid forward pass.                                       #
    #############################################################################
    out = sigmoid(x)
    #############################################################################
    #                                                                           #
    #############################################################################
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of sigmoid units.

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Values from the forward pass, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, sigmoid_x = None, cache
    #############################################################################
    # Implement the ReLU backward pass.                                         #
    #############################################################################
    dx = sigmoid_x * (1 - sigmoid_x) * dout
    #############################################################################
    #                                                                           #
    #############################################################################
    return dx


def lstm_forward_unit(x, w, u, b, h_prev=None, c_prev=None):
    """
    Computes the forward pass for a single time step of a long short term memory layer.
    ref: https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of input weights, of shape (D, 4M)
    - u: A numpy array of state weights, of shape (M, 4M)
    - b: A numpy array of biases, of shape (4M,)
    - h_prev: A numpy array containing hidden state generated by previous step, of shape (N, M)
    - c_prev: A numpy array containing cell state generated by previous step, of shape (N, M)

    Returns a tuple of:
    - h: new hidden state, of shape (N, M)
    - c: new cell state, of shape (N, M)
    - cache: cache for back propagation, (x, h_prev, c_prev, tanh_c, g, w, u)
      - g: A numpy array of gates, of shape (N, 4M)
    """
    N = np.shape(x)[0]
    M = np.shape(u)[0]
    h_prev = np.zeros((N, M)) if h_prev is None else h_prev
    c_prev = np.zeros((N, M)) if c_prev is None else c_prev
    h, c = None, None
    #############################################################################
    # Implement the LSTM forward pass unit. Store the result in out.            #
    # Will need to reshape the input into rows.                                 #
    #############################################################################
    z = x.reshape(np.shape(x)[0], -1)  # N*D

    mid = np.dot(z, w) + np.dot(h_prev, u) + b  # N*4M
    a = tanh(mid[:, :M])  # N*M
    i = sigmoid(mid[:, M:(2*M)])  # N*M
    f = sigmoid(mid[:, (2*M):(3*M)])  # N*M
    o = sigmoid(mid[:, (3*M):(4*M)])  # N*M
    g = np.concatenate((a, i, f, o), axis=1)  # N*4M

    c = a * i + f * c_prev  # N*M
    tanh_c = tanh(c)
    h = tanh_c * o  # N*M
    #############################################################################
    #                                                                           #
    #############################################################################
    cache = (x, h_prev, c_prev, tanh_c, g, w, u)
    return h, c, cache


def lstm_backward_unit(dout, cache, dh=None, dc_next=None, f_next=None):
    """
    Computes the backward pass for a LSTM unit.
    ref: https://blog.aidangomez.ca/2016/04/17/Backpropogating-an-LSTM-A-Numerical-Example/

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of x, h_prev, c_prev, tanh_c, g, w, u
    - dh: derivative of hidden state passed from the next step, of shape (N, M)
    - dc_next: derivative of cell state of the next step, of shape (N, M)
    - f_next: value of forget gate of the next step, of shape (N, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, 4M)
    - du: Gradient with respect to u, of shape (M, 4M)
    - db: Gradient with respect to b, of shape (4M,)
    - dh_prev, dc, f: Time cache to be passed to the previous step of propagation. Correspond to dh, dc_next, f_next
    """
    x, h_prev, c_prev, tanh_c, g, w, u = cache
    N, M = np.shape(h_prev)
    a = g[:, :M]
    i = g[:, M:(2*M)]
    f = g[:, (2*M):(3*M)]
    o = g[:, (3*M):(4*M)]

    dh = np.zeros((N, M)) if dh is None else dh
    dc_next = np.zeros((N, M)) if dc_next is None else dc_next
    f_next = np.zeros((N, M)) if f_next is None else f_next
    dx, dw, du, db = None, None, None, None
    #############################################################################
    # Implement the affine backward pass.                                       #
    #############################################################################
    dy = dout + dh  # N*M
    dc = dy * o * (1 - tanh_c ** 2) + dc_next * f_next  # N*M

    da = dc * i * (1 - a ** 2)
    di = dc * a * i * (1 - i)
    df = dc * c_prev * f * (1 - f)
    do = dy * tanh_c * o * (1 - o)
    dg = np.concatenate((da, di, df, do), axis=1)  # N*4M

    shapes = np.shape(x)
    z = x.reshape(N, -1)

    dx = np.dot(dg, w.T).reshape(shapes)  # N*d1*...*dk
    dh_prev = np.dot(dg, u.T)  # N*M

    dw = np.dot(z.T, dg)  # D*4M
    du = np.dot(h_prev.T, dg)  # M*4M
    db = np.dot(np.ones(N), dg)  # 4M
    # Note: Gradient here is the sum of the gradients of N data
    #############################################################################
    #                                                                           #
    #############################################################################
    # update: dw, du, db
    # time cache: dh_prev, dc, f
    # space cache: dx
    return dx, dw, du, db, dh_prev, dc, f


""" loss """
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


# new loss
def mse_loss(x, y, w=None):
    """
    Computes the loss and gradient for mean squared error regression.

    Inputs:
    - x: Input data, of shape (N, F) where x[i, j] is the predicted value for the jth dimension
      of the ith input.
    - y: Vector of true outputs, of shape (N, F) where y[i] is the label for x[i] and
      0 <= y[i] < C
    - w: Weight of features, of shape (D, )

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N, F = np.shape(x)
    if w is None:
        w = np.ones(F)
    err = x - y  # N*F
    loss = np.mean(np.dot(err ** 2, w / F))
    dx = 2./F * w ** 0.5 * err
    dx /= N
    return loss, dx

