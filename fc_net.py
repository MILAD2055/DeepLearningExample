import os
import numpy as np

from .layers import (
    affine_forward,
    affine_backward,
    relu_forward,
    relu_backward,
    batchnorm_forward,
    batchnorm_backward_alt,
    layernorm_forward,
    layernorm_backward,
    dropout_forward,
    dropout_backward,
    softmax_loss,
)
from .layer_utils import affine_relu_forward, affine_relu_backward


class TwoLayerNet(object):

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # Step 2.3 : Initialize Parameters
        # W1: (input_dim, hidden_dim), b1: (hidden_dim,)
        # W2: (hidden_dim, num_classes), b2: (num_classes,)
        ############################################################################
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b2"] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
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
        # Step 4.2 : Hidden Layer Forward -> affine + ReLU
        # Step 4.3 : Final Output Layer -> affine only (no activation, raw scores)
        ############################################################################
        hidden_out, hidden_cache = affine_relu_forward(
            X, self.params["W1"], self.params["b1"]
        )
        scores, out_cache = affine_forward(
            hidden_out, self.params["W2"], self.params["b2"]
        )
        # Step 4.4 : Scores -> `scores` above, shape (N, num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # Step 5 : Loss Calculation
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)  # Step 5.1-5.3
        reg_loss = 0.5 * self.reg * (
            np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)
        )  # Step 5.4
        loss = data_loss + reg_loss  # Step 5.5 -> dscores is dout, ready for backward

        ############################################################################
        # Step 6 : Backward Pass & Gradient
        ############################################################################
        # Step 6.2 : Final Affine Backward
        dhidden, dW2, db2 = affine_backward(dscores, out_cache)
        dW2 += self.reg * self.params["W2"]  # add regularization gradient

        # Step 6.3 : Backward through hidden layer (ReLU then affine)
        dX, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)
        dW1 += self.reg * self.params["W1"]

        # Step 6.4 : All Gradients Ready
        grads["W1"], grads["b1"] = dW1, db1
        grads["W2"], grads["b2"] = dW2, db2
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # Step 2.2 : Create Layer Dimensions
        # dims = [input_dim, hidden_dims..., num_classes] so consecutive pairs
        # give each affine layer's (in, out) shape.
        ############################################################################
        dims = [input_dim] + list(hidden_dims) + [num_classes]

        ############################################################################
        # Step 2.3 : Initialize Parameters -> weights, biases, and (if used)
        # gamma/beta for every layer, all in the same routine.
        ############################################################################
        for i in range(self.num_layers):
            self.params["W%d" % (i + 1)] = weight_scale * np.random.randn(
                dims[i], dims[i + 1]
            )
            self.params["b%d" % (i + 1)] = np.zeros(dims[i + 1])

            # Norm params only exist on hidden layers (not the final output layer)
            if self.normalization in ("batchnorm", "layernorm") and i < self.num_layers - 1:
                self.params["gamma%d" % (i + 1)] = np.ones(dims[i + 1])
                self.params["beta%d" % (i + 1)] = np.zeros(dims[i + 1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # Step 4.1 : Prepare Input -> `out` starts as X
        # Step 4.2 : Hidden Layer Forward, repeated for layers 1 .. num_layers-1
        #            affine -> [norm] -> relu -> [dropout]
        # Step 4.3 : Final Output Layer -> affine only, no relu/norm/dropout
        # Step 4.4 : Scores
        ############################################################################
        caches = {}
        out = X
        for i in range(1, self.num_layers):
            W, b = self.params["W%d" % i], self.params["b%d" % i]
            a, affine_cache = affine_forward(out, W, b)

            norm_cache = None
            if self.normalization == "batchnorm":
                gamma, beta = self.params["gamma%d" % i], self.params["beta%d" % i]
                a, norm_cache = batchnorm_forward(a, gamma, beta, self.bn_params[i - 1])
            elif self.normalization == "layernorm":
                gamma, beta = self.params["gamma%d" % i], self.params["beta%d" % i]
                a, norm_cache = layernorm_forward(a, gamma, beta, self.bn_params[i - 1])

            relu_out, relu_cache = relu_forward(a)

            dropout_cache = None
            if self.use_dropout:
                relu_out, dropout_cache = dropout_forward(relu_out, self.dropout_param)

            caches[i] = (affine_cache, norm_cache, relu_cache, dropout_cache)
            out = relu_out

        # Final output layer (Step 4.3) — no norm/relu/dropout
        W_last, b_last = self.params["W%d" % self.num_layers], self.params["b%d" % self.num_layers]
        scores, final_cache = affine_forward(out, W_last, b_last)
        caches["final"] = final_cache
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # Step 5 : Loss Calculation
        ############################################################################
        data_loss, dscores = softmax_loss(scores, y)  # Step 5.1-5.3
        reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            reg_loss += 0.5 * self.reg * np.sum(self.params["W%d" % i] ** 2)  # Step 5.4
        loss = data_loss + reg_loss  # Step 5.5 -> dscores is dout, ready for backward

        ############################################################################
        # Step 6 : Backward Pass & Gradient
        ############################################################################
        # Step 6.2 : Final Affine Backward
        dout, dW_last, db_last = affine_backward(dscores, caches["final"])
        dW_last += self.reg * self.params["W%d" % self.num_layers]
        grads["W%d" % self.num_layers] = dW_last
        grads["b%d" % self.num_layers] = db_last

        # Step 6.3 : Backward through hidden layers, in reverse order
        for i in range(self.num_layers - 1, 0, -1):
            affine_cache, norm_cache, relu_cache, dropout_cache = caches[i]

            if self.use_dropout:
                dout = dropout_backward(dout, dropout_cache)

            dout = relu_backward(dout, relu_cache)

            if self.normalization == "batchnorm":
                dout, dgamma, dbeta = batchnorm_backward_alt(dout, norm_cache)
                grads["gamma%d" % i] = dgamma
                grads["beta%d" % i] = dbeta
            elif self.normalization == "layernorm":
                dout, dgamma, dbeta = layernorm_backward(dout, norm_cache)
                grads["gamma%d" % i] = dgamma
                grads["beta%d" % i] = dbeta

            dout, dW, db = affine_backward(dout, affine_cache)
            dW += self.reg * self.params["W%d" % i]
            grads["W%d" % i] = dW
            grads["b%d" % i] = db

        # Step 6.4 : All Gradients Ready -> return loss, grads
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True