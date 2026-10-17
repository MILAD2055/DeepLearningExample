# -*- coding: utf-8 -*-
"""
fully_connected_net_all_in_one.py
=================================

All files of the cs231nMe package + the FullyConnectedNet notebook, merged into
ONE Python file. Each original file is kept as its own clearly marked PART.

    PART 1 : layers.py          -> forward / backward of every layer + softmax loss
    PART 2 : layer_utils.py     -> affine_relu "sandwich" layer
    PART 3 : optim.py           -> update rules: sgd, sgd_momentum, rmsprop, adam
    PART 4 : fc_net.py          -> TwoLayerNet and FullyConnectedNet models
    PART 5 : solver.py          -> Solver: training loop + ACCURACY (check_accuracy)
    PART 6 : data_utils.py      -> loading CIFAR-10 (and other datasets)
    PART 7 : gradient_check.py  -> numerical gradient checking tools
    PART 8 : FullyConnectedNet.ipynb (MAIN) -> Steps 1..7 of the pipeline
             + Step 8 (test accuracy) taken from pipeline.py

فهرست بخش‌ها:
    بخش ۱ تا ۷  = فایل‌های پکیج cs231nMe (تعریف توابع و کلاس‌ها، چیزی اجرا نمی‌کنند)
    بخش ۸       = نوت‌بوک اصلی (همان چیزی که اجرا می‌شود)

Changes needed to make it ONE file (behaviour is the same):
  - Relative imports (`from .layers import ...`, `from . import optim`) removed,
    because everything now lives in the same file.
  - Solver finds the update rule from the OPTIM_RULES dict instead of the optim module.
  - Dataset / save folders are set in CONFIG below instead of `__file__/datasets`.
  - Colab shell commands (!pwd, !wget, %cd) replaced with plain Python.
"""

import os
import sys
import time
import pickle
import platform
import tarfile
import urllib.request
from random import randrange

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG  (تنظیمات مسیرها)
# =============================================================================
# On Colab the data lives on Google Drive; anywhere else, next to this file.
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    PROJECT_DIR = "/content/drive/My Drive/cs231nMe"
else:
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
CIFAR10_DIR = os.path.join(DATASET_DIR, "cifar-10-batches-py")
SAVE_DIR = os.path.join(PROJECT_DIR, "saved")


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 1 : layers.py                                                    ##
# ##   بخش ۱ : لایه‌ها (affine, relu, batchnorm, layernorm, dropout, loss)   ##
# ##                                                                         ##
# #############################################################################
# #############################################################################


# -----------------------------------------------------------------------------
# 1.1  Affine (fully-connected) layer
# -----------------------------------------------------------------------------
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
    # Reshape each input example into a single row of length D, then apply
    # the affine transform: out = X_flat @ W + b
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    out = x_flat.dot(w) + b
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
    # dx = dout @ W^T, reshaped back to x's original shape
    # dW = X_flat^T @ dout
    # db = sum of dout over the batch
    N = x.shape[0]
    x_flat = x.reshape(N, -1)
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x_flat.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


# -----------------------------------------------------------------------------
# 1.2  ReLU
# -----------------------------------------------------------------------------
def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    # ReLU: elementwise max(0, x)
    out = np.maximum(0, x)
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
    x = cache
    # Gradient only flows through units that were positive in the forward pass
    dx = dout * (x > 0)
    return dx


# -----------------------------------------------------------------------------
# 1.3  Batch normalization
# -----------------------------------------------------------------------------
def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

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
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        # Training: normalize with THIS mini-batch's mean / variance,
        # then scale (gamma) and shift (beta). Also update running averages.
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        std = np.sqrt(sample_var + eps)
        x_norm = (x - sample_mean) / std
        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        cache = (x, x_norm, sample_mean, sample_var, std, gamma, beta, eps)
    elif mode == "test":
        # Test: normalize with the RUNNING mean / variance from training.
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization (step-by-step computation graph).

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, x_norm, sample_mean, sample_var, std, gamma, beta, eps = cache
    N = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    dx_norm = dout * gamma
    dvar = np.sum(dx_norm * (x - sample_mean) * -0.5 * std ** -3, axis=0)
    dmean = np.sum(dx_norm * -1 / std, axis=0) + dvar * np.mean(
        -2 * (x - sample_mean), axis=0
    )
    dx = dx_norm / std + dvar * 2 * (x - sample_mean) / N + dmean / N

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization
    (simplified closed-form version of batchnorm_backward).

    Inputs / outputs: Same as batchnorm_backward
    """
    x, x_norm, sample_mean, sample_var, std, gamma, beta, eps = cache
    N = dout.shape[0]

    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    dx_norm = dout * gamma
    dx = (1.0 / N) / std * (
        N * dx_norm - np.sum(dx_norm, axis=0) - x_norm * np.sum(dx_norm * x_norm, axis=0)
    )

    return dx, dgamma, dbeta


# -----------------------------------------------------------------------------
# 1.4  Layer normalization
# -----------------------------------------------------------------------------
def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    eps = ln_param.get("eps", 1e-5)
    # Layer norm normalizes per-example (across features) instead of
    # per-feature (across the batch). Transposing turns it into the same
    # computation as batchnorm on x.T, then we transpose the result back.
    xT = x.T
    mean = np.mean(xT, axis=0)
    var = np.var(xT, axis=0)
    std = np.sqrt(var + eps)
    x_norm_T = (xT - mean) / std
    x_norm = x_norm_T.T
    out = gamma * x_norm + beta

    cache = (x, x_norm, mean, var, std, gamma, beta, eps)
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, x_norm, mean, var, std, gamma, beta, eps = cache
    D = dout.shape[1]

    # gamma/beta gradients are still per-feature, so those sums stay over axis 0.
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_norm, axis=0)

    # For dx, reuse the batchnorm_backward_alt formula on the transposed layout.
    dx_norm_T = (dout * gamma).T
    x_norm_T = x_norm.T
    dxT = (1.0 / D) / std * (
        D * dx_norm_T
        - np.sum(dx_norm_T, axis=0)
        - x_norm_T * np.sum(dx_norm_T * x_norm_T, axis=0)
    )
    dx = dxT.T
    return dx, dgamma, dbeta


# -----------------------------------------------------------------------------
# 1.5  Dropout (inverted dropout)
# -----------------------------------------------------------------------------
def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator (for gradient checking).

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask).

    NOTE: p is the probability of **keeping** a neuron output.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        # Inverted dropout: scale by 1/p at train time so no rescaling is
        # needed at test time.
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        # Test-time dropout is a no-op — the scaling already happened in train
        out = x

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
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        # Gradient only flows through units that survived the mask, scaled
        # the same way the forward pass scaled them.
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx


# -----------------------------------------------------------------------------
# 1.6  Convolution / pooling / spatial norms
#      NOTE: still empty TODO stubs in the original (CNN assignment, not used here)
# -----------------------------------------------------------------------------
def conv_forward_naive(x, w, b, conv_param):
    """Naive convolution forward pass. (TODO in original — not implemented.)"""
    out = None
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """Naive convolution backward pass. (TODO in original — not implemented.)"""
    dx, dw, db = None, None, None
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """Naive max-pool forward pass. (TODO in original — not implemented.)"""
    out = None
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """Naive max-pool backward pass. (TODO in original — not implemented.)"""
    dx = None
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Spatial batchnorm forward. (TODO in original — not implemented.)"""
    out, cache = None, None
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Spatial batchnorm backward. (TODO in original — not implemented.)"""
    dx, dgamma, dbeta = None, None, None
    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Spatial groupnorm forward. (TODO in original — not implemented.)"""
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)  # noqa: F841
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Spatial groupnorm backward. (TODO in original — not implemented.)"""
    dx, dgamma, dbeta = None, None, None
    return dx, dgamma, dbeta


# -----------------------------------------------------------------------------
# 1.7  Loss functions: SVM and Softmax
# -----------------------------------------------------------------------------
def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # Multiclass (Weston-Watkins) hinge loss: margin = max(0, other - correct + 1)
    N = x.shape[0]
    correct_scores = x[np.arange(N), y].reshape(-1, 1)
    margins = np.maximum(0, x - correct_scores + 1.0)
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
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # Step 5.1: numerical stability -> subtract max score per row, exponentiate
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)

    # Step 5.1 (cont.): normalize -> class probabilities
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = x.shape[0]

    # Step 5.2: select the probability assigned to the correct class per row
    correct_class_probs = probs[np.arange(N), y]

    # Step 5.3: data loss -> -log(correct-class prob), averaged over the batch
    loss = -np.sum(np.log(correct_class_probs)) / N

    # dx (dout) for the backward pass: softmax-cross-entropy gradient is
    # (probs - 1) for the correct class, probs elsewhere, averaged over N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 2 : layer_utils.py                                               ##
# ##   بخش ۲ : لایهٔ ترکیبی affine + relu                                    ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 3 : optim.py                                                     ##
# ##   بخش ۳ : قوانین به‌روزرسانی پارامترها (SGD, Momentum, RMSProp, Adam)   ##
# ##                                                                         ##
# #############################################################################
# #############################################################################
"""
Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    # v = momentum * v - learning_rate * dw   (moving average of gradients)
    # next_w = w + v
    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w + v
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    # cache = decay_rate * cache + (1 - decay_rate) * dw^2
    # next_w = w - learning_rate * dw / (sqrt(cache) + epsilon)
    cache = config["cache"]
    cache = config["decay_rate"] * cache + (1 - config["decay_rate"]) * (dw ** 2)
    next_w = w - config["learning_rate"] * dw / (np.sqrt(cache) + config["epsilon"])
    config["cache"] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    # t += 1
    # m = beta1*m + (1-beta1)*dw            -> moving average of gradient
    # v = beta2*v + (1-beta2)*dw^2          -> moving average of squared grad
    # m_hat, v_hat -> bias-corrected using t
    # next_w = w - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
    config["t"] += 1
    t = config["t"]
    beta1, beta2, eps = config["beta1"], config["beta2"], config["epsilon"]

    m = beta1 * config["m"] + (1 - beta1) * dw
    v = beta2 * config["v"] + (1 - beta2) * (dw ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    next_w = w - config["learning_rate"] * m_hat / (np.sqrt(v_hat) + eps)

    config["m"] = m
    config["v"] = v

    return next_w, config


# Name -> function lookup, used by Solver (replaces `getattr(optim, name)`).
OPTIM_RULES = {
    "sgd": sgd,
    "sgd_momentum": sgd_momentum,
    "rmsprop": rmsprop,
    "adam": adam,
}


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 4 : fc_net.py                                                    ##
# ##   بخش ۴ : مدل‌ها (TwoLayerNet و FullyConnectedNet)                       ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

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

        # Step 2.3 : Initialize Parameters
        # W1: (input_dim, hidden_dim), b1: (hidden_dim,)
        # W2: (hidden_dim, num_classes), b2: (num_classes,)
        self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params["b1"] = np.zeros(hidden_dim)
        self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b2"] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params.
        """
        # Step 4.2 : Hidden Layer Forward -> affine + ReLU
        # Step 4.3 : Final Output Layer -> affine only (no activation, raw scores)
        hidden_out, hidden_cache = affine_relu_forward(
            X, self.params["W1"], self.params["b1"]
        )
        scores, out_cache = affine_forward(
            hidden_out, self.params["W2"], self.params["b2"]
        )
        # Step 4.4 : Scores -> `scores` above, shape (N, num_classes)

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        # Step 5 : Loss Calculation
        data_loss, dscores = softmax_loss(scores, y)  # Step 5.1-5.3
        reg_loss = 0.5 * self.reg * (
            np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)
        )  # Step 5.4
        loss = data_loss + reg_loss  # Step 5.5 -> dscores is dout, ready for backward

        # Step 6 : Backward Pass & Gradient
        # Step 6.2 : Final Affine Backward
        dhidden, dW2, db2 = affine_backward(dscores, out_cache)
        dW2 += self.reg * self.params["W2"]  # add regularization gradient

        # Step 6.3 : Backward through hidden layer (ReLU then affine)
        dX, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)
        dW1 += self.reg * self.params["W1"]

        # Step 6.4 : All Gradients Ready
        grads["W1"], grads["b1"] = dW1, db1
        grads["W2"], grads["b2"] = dW2, db2

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        fpath = os.path.join(SAVE_DIR, fname)
        np.save(fpath, self.params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(SAVE_DIR, fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
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
        - normalization: "batchnorm", "layernorm", or None (default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float64 is needed for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # Step 2.2 : Create Layer Dimensions
        # dims = [input_dim, hidden_dims..., num_classes] so consecutive pairs
        # give each affine layer's (in, out) shape.
        dims = [input_dim] + list(hidden_dims) + [num_classes]

        # Step 2.3 : Initialize Parameters -> weights, biases, and (if used)
        # gamma/beta for every layer, all in the same routine.
        for i in range(self.num_layers):
            self.params["W%d" % (i + 1)] = weight_scale * np.random.randn(
                dims[i], dims[i + 1]
            )
            self.params["b%d" % (i + 1)] = np.zeros(dims[i + 1])

            # Norm params only exist on hidden layers (not the final output layer)
            if self.normalization in ("batchnorm", "layernorm") and i < self.num_layers - 1:
                self.params["gamma%d" % (i + 1)] = np.ones(dims[i + 1])
                self.params["beta%d" % (i + 1)] = np.zeros(dims[i + 1])

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
        # normalization layer.
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
        If y is None -> test-time forward pass, returns scores (N, C).
            (This is what Solver.check_accuracy uses to compute ACCURACY.)
        If y is not None -> training-time forward + backward pass, returns (loss, grads).
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

        # Step 4.1 : Prepare Input -> `out` starts as X
        # Step 4.2 : Hidden Layer Forward, repeated for layers 1 .. num_layers-1
        #            affine -> [norm] -> relu -> [dropout]
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

        # Step 4.3 : Final Output Layer — no norm/relu/dropout
        W_last, b_last = self.params["W%d" % self.num_layers], self.params["b%d" % self.num_layers]
        scores, final_cache = affine_forward(out, W_last, b_last)
        caches["final"] = final_cache
        # Step 4.4 : Scores

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        # Step 5 : Loss Calculation
        data_loss, dscores = softmax_loss(scores, y)  # Step 5.1-5.3
        reg_loss = 0.0
        for i in range(1, self.num_layers + 1):
            reg_loss += 0.5 * self.reg * np.sum(self.params["W%d" % i] ** 2)  # Step 5.4
        loss = data_loss + reg_loss  # Step 5.5 -> dscores is dout, ready for backward

        # Step 6 : Backward Pass & Gradient
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
        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        os.makedirs(SAVE_DIR, exist_ok=True)
        fpath = os.path.join(SAVE_DIR, fname)
        np.save(fpath, self.params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(SAVE_DIR, fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 5 : solver.py                                                    ##
# ##   بخش ۵ : Solver — حلقهٔ آموزش و محاسبهٔ ACCURACY                        ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in PART 3 (optim).

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, solver.loss_history contains all losses during training and
    solver.train_acc_history / solver.val_acc_history contain the accuracies
    on the training and validation set at each epoch.
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object (params dict + loss(X, y) method)
        - data: A dictionary with 'X_train', 'y_train', 'X_val', 'y_val'

        Optional arguments:
        - update_rule: Name of an update rule in OPTIM_RULES. Default is 'sgd'.
        - optim_config: Hyperparameters passed to the update rule
          (must contain 'learning_rate').
        - lr_decay: After each epoch the learning rate is multiplied by this value.
        - batch_size: Size of minibatches.
        - num_epochs: Number of epochs to train.
        - print_every: Print training loss every print_every iterations.
        - verbose: If False, print nothing.
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; None = entire training set.
        - num_val_samples: Number of validation samples used to check val
          accuracy; default is None = entire validation set.
        - checkpoint_name: If not None, save model checkpoints here every epoch.
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        if self.update_rule not in OPTIM_RULES:
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = OPTIM_RULES[self.update_rule]

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update (Steps 3.4 -> 7.4). Called by train().
        """
        # Step 3.4 : Make a minibatch of training data
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Steps 4-6 : Compute loss and gradient
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Step 7 : Perform a parameter update
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    # -------------------------------------------------------------------------
    # >>> ACCURACY IS COMPUTED HERE <<<      >>> دقت اینجا حساب می‌شود <<<
    # -------------------------------------------------------------------------
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])        # test mode (y=None) -> scores
            y_pred.append(np.argmax(scores, axis=1))      # predicted class = highest score
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)                        # fraction of correct predictions

        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Step 8.1 / 8.2 : Check train and val ACCURACY on the first iteration,
            # the last iteration, and at the end of each epoch.
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # Keep track of the best model (highest validation accuracy)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 6 : data_utils.py                                                ##
# ##   بخش ۶ : بارگذاری داده (CIFAR-10 و ...)                                ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

try:
    from imageio import imread
except ImportError:
    imread = None  # only needed by load_tiny_imagenet(), not by get_CIFAR10_data()


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers.
    """
    # Load the raw CIFAR-10 data (folder set in CONFIG at the top of this file)
    cifar10_dir = CIFAR10_DIR
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """Load TinyImageNet (not used in this project)."""
    # First load wnids
    with open(os.path.join(path, "wnids.txt"), "r") as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, "words.txt"), "r") as f:
        wnid_to_words = dict(line.split("\t") for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(",")]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print("loading training data for synset %d / %d" % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, "train", wnid, "%s_boxes.txt" % wnid)
        with open(boxes_file, "r") as f:
            filenames = [x.split("\t")[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, "train", wnid, "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                ## grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, "val", "val_annotations.txt"), "r") as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split("\t")[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, "val", "images", img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    img_files = os.listdir(os.path.join(path, "test", "images"))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, "test", "images", img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, "test", "test_annotations.txt")
    if os.path.isfile(y_test_file):
        with open(y_test_file, "r") as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split("\t")
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
        "class_names": class_names,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "mean_image": mean_image,
    }


def load_models(models_dir):
    """
    Load saved models from disk. Any files that give errors on unpickling
    (such as README.txt) will be skipped.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), "rb") as f:
            try:
                models[model_file] = load_pickle(f)["model"]
            except pickle.UnpicklingError:
                continue
    return models


def load_imagenet_val(num=None):
    """Load a handful of validation images from ImageNet (not used in this project)."""
    imagenet_fn = os.path.join(DATASET_DIR, "imagenet_val_25.npz")
    if not os.path.isfile(imagenet_fn):
        print("file %s not found" % imagenet_fn)
        print("Run the following:")
        print("cd cs231n/datasets")
        print("bash get_imagenet_val.sh")
        assert False, "Need to download imagenet_val_25.npz"

    f = np.load(imagenet_fn, allow_pickle=True)
    X = f["X"]
    y = f["y"]
    class_names = f["label_map"].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 7 : gradient_check.py                                            ##
# ##   بخش ۷ : بررسی عددی گرادیان                                            ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """
    fx = f(x)  # evaluate function value at original point  # noqa: F841
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.

    We assume that f accepts several input blobs as arguments, followed by a
    blob where outputs will be written.
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )


# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART 8 : FullyConnectedNet.ipynb  (MAIN — this is what runs)          ##
# ##   بخش ۸ : نوت‌بوک اصلی — اجرای Step 1 تا Step 8                          ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

if __name__ == "__main__":

    # =========================================================================
    # Step 1 : Setup & Data Preparation
    # =========================================================================

    # Step 1.1 : Environment Setup & Import tools
    # (numpy / matplotlib already imported at the top of the file;
    #  `from cs231nMe import *` is no longer needed — PARTS 1-7 are above.)
    if IN_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')

    plt.rcParams["figure.figsize"] = (10.0, 8.0)
    plt.rcParams["image.interpolation"] = "nearest"
    plt.rcParams["image.cmap"] = "gray"

    # Step 1.3 : Download Dataset - CIFAR10
    # Only download if it's not already there — safe to re-run
    os.makedirs(DATASET_DIR, exist_ok=True)
    if not os.path.exists(CIFAR10_DIR):
        tar_path = os.path.join(DATASET_DIR, "cifar-10-python.tar.gz")
        print("Downloading CIFAR-10 ...")
        urllib.request.urlretrieve(
            "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tar_path
        )
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(DATASET_DIR)
        print("Downloaded and extracted CIFAR-10.")
    else:
        print("CIFAR-10 already present, skipping download.")

    # Step 1.4 : Load Dataset
    data = get_CIFAR10_data()
    for k, v in list(data.items()):
        print(f"{k}: {v.shape}")

    # Step 1.5 : Prepare & Create Data -> reshape + mean-subtraction normalization,
    # then package the split subsets the solver will train/validate on.
    X_train_vectorized = data["X_train"].reshape(data["X_train"].shape[0], -1)
    X_val_vectorized = data["X_val"].reshape(data["X_val"].shape[0], -1)
    X_test_vectorized = data["X_test"].reshape(data["X_test"].shape[0], -1)

    mean_image = np.mean(X_train_vectorized, axis=0)
    X_train_vectorized -= mean_image
    X_val_vectorized -= mean_image
    X_test_vectorized -= mean_image

    # A small part for FUNCTIONALITY TEST like overfitting and etc tests befor a full training run.
    num_train_small = 50

    small_data = {
        "X_train": X_train_vectorized[:num_train_small],
        "y_train": data["y_train"][:num_train_small],
        "X_val": X_val_vectorized,
        "y_val": data["y_val"],
    }

    full_data = {
        "X_train": X_train_vectorized,
        "y_train": data["y_train"],
        "X_val": X_val_vectorized,
        "y_val": data["y_val"],
    }

    # =========================================================================
    # Step 2 : Model Creation & Initialization  (runs once)
    # =========================================================================

    # Step 2.1 & 2.2 : Define Architecture & Create Layer Dimensions
    input_dim = X_train_vectorized.shape[1]  # num of features = pixels in 1 image (3 * 32 * 32 = 3072)
    hidden_dims = [100, 100]
    num_classes = 10

    # Step 2.3 : Initialize Parameters (W, b, and gamma/beta if normalization is used)
    # -- happens inside FullyConnectedNet.__init__ (PART 4) when the model object is built:
    model = FullyConnectedNet(
        hidden_dims,
        input_dim=input_dim,
        num_classes=num_classes,
        reg=1e-2,  # total loss = how wrong + reg × (size of weights)
        weight_scale=5e-2,
        dtype=np.float64,
        normalization="batchnorm",   # set to None to disable
        dropout_keep_ratio=0.9,      # set to 1 to disable dropout
    )
    # Initialization => b = b - learning_rate × db

    # =========================================================================
    # Step 3 : Training Setup & Mini-Batch
    # =========================================================================

    # Step 3.1 : Select training data
    # We can change full_data to small_data to check overfitting
    train_data = full_data

    # Step 3.2 : Training Hyperparameters
    learning_rate = 1e-3
    batch_size = 100
    num_epochs = 10

    # Step 3.3 : Create Solver (PART 5) -> wraps model + data + hyperparameters together.
    # This does NOT rebuild the model — it reuses the one created in Step 2.
    solver = Solver(
        model,
        train_data,
        update_rule="adam",
        optim_config={"learning_rate": learning_rate},
        lr_decay=0.95,
        batch_size=batch_size,
        num_epochs=num_epochs,
        print_every=100,
    )

    # Step 3.4 : Start Mini-Batch  <- loop begins here, repeats each iteration
    # Calling solver.train() runs Steps 3.4 -> 7.4 in a loop internally:
    # sample mini-batch -> forward pass -> loss -> backward pass -> optimizer
    # update -> back to 3.4, until num_epochs is reached.
    # ACCURACY: train()  calls  check_accuracy() (PART 5) at the end of every epoch.
    solver.train()

    # =========================================================================
    # Step 8 : Evaluation (ACCURACY)       مرحلهٔ ۸ : ارزیابی و دقت
    # =========================================================================

    # Step 8.1 : Plot train / val accuracy recorded by solver.train()
    plt.plot(solver.train_acc_history, label='train accuracy')
    plt.plot(solver.val_acc_history, label='val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Train vs Val Accuracy')
    plt.show()

    # Step 8.2 / 8.3 : Final test accuracy  (from pipeline.py)
    # Must run HERE — before the manual Steps 4-7 below, which change the model.
    test_acc = solver.check_accuracy(X_test_vectorized, data["y_test"])
    print("Final test accuracy: %f" % test_acc)
    print("Best validation accuracy: %f" % solver.best_val_acc)

    # =========================================================================
    # Steps 4 -> 7 : ONE training iteration done BY HAND (to show what happens
    # inside solver.train()).    یک تکرار آموزش به‌صورت دستی
    # =========================================================================

    # --- Step 4 setup: real mini-batch, model in train mode ---

    np.random.seed(231)

    # among all number of images (train_data["X_train"].shape[0]) choose number in batch size RANDOMLY.
    batch_mask = np.random.choice(train_data["X_train"].shape[0], batch_size)

    # From dic of train_data choose all images of training, then choose among them in number of batch_mask
    X = train_data["X_train"][batch_mask]
    y = train_data["y_train"][batch_mask]

    # repeats once per hidden layer
    # From the first iteration, we turn off some neurons randomly (dropout).
    # Each iteration, we change them and turn off a different random group,
    # so no single neuron becomes too dependent on the others.
    model.dropout_param["mode"] = "train"

    # For each hidden layer we have a batch normalization, so we change the mode of all layers to training
    # beacause in training mode we use mean of that batch not all of batch
    for bn_param in model.bn_params:
        bn_param["mode"] = "train"

    # -------------------------------------------------------------------------
    # Step 4 : Forward Pass (manual/unrolled version — matches FullyConnectedNet.loss())
    # -------------------------------------------------------------------------
    # Step 4.1 : Prepare Input
    # Copy X into a new variable "out"
    out = X
    affine_caches, norm_caches, relu_caches, dropout_caches = [], [], [], []

    # Step 4.2 : Hidden Layer Forward -- affine -> batchnorm -> relu -> dropout,
    # repeats once per hidden layer
    for i in range(1, model.num_layers):
        W = model.params[f"W{i}"]
        b = model.params[f"b{i}"]

        a, affine_cache = affine_forward(out, W, b)
        affine_caches.append(affine_cache)
        # ___________________________________________________
        gamma = model.params[f"gamma{i}"]
        beta = model.params[f"beta{i}"]

        a, norm_cache = batchnorm_forward(a, gamma, beta, model.bn_params[i - 1])
        norm_caches.append(norm_cache)
        # ___________________________________________________
        relu_out, relu_cache = relu_forward(a)
        relu_caches.append(relu_cache)
        # ___________________________________________________
        relu_out, dropout_cache = dropout_forward(relu_out, model.dropout_param)
        dropout_caches.append(dropout_cache)

        out = relu_out
        print(f"  hidden layer {i} output shape: {out.shape}")

    # Step 4.3 : Final Output Layer -- affine only, no relu/norm/dropout
    W_last = model.params[f"W{model.num_layers}"]
    b_last = model.params[f"b{model.num_layers}"]
    scores, final_cache = affine_forward(out, W_last, b_last)

    # Step 4.4 : Scores
    print("Final scores shape:", scores.shape)

    # -------------------------------------------------------------------------
    # Step 5 : Loss Calculation
    # -------------------------------------------------------------------------

    # Step 5.1 : Softmax(scores)
    # numerical stability => each score - max score    # keepdims=True => to make -/+ possible
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)

    # Exp => we have smaller scores and some negative scores => exp(-scores) is positive and small
    exp_scores = np.exp(shifted_scores)

    # class probabilities
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    print("Probabilities shape:", probs.shape)   # (N, C)

    # Sum of all probs for each image and sum = 1 --> axis = 1 --> add across the row (left to right)
    # --> axis=0 would mean "add down the column"
    print("Each row sums to 1:", np.sum(probs, axis=1))

    # Step 5.2 : Select Correct-Class Probabilities
    N = scores.shape[0]

    # first --> go to a row --> inside it, print the real value of correct class
    correct_class_probs = probs[np.arange(N), y]
    print("Correct-class probabilities:", correct_class_probs)

    # Step 5.3 => data loss : sum all correct class --> -log () --> /N
    # CrossEntropic => for classification
    # MSE => Regression
    data_loss = -np.sum(np.log(correct_class_probs)) / N
    print("Data loss:", data_loss)

    # Step 5.4 : L2 Regularization Loss
    # 0.5 * reg * sum(W^2), summed across every weight matrix in the network
    reg_loss = 0.0
    for i in range(1, model.num_layers + 1):
        # get the matrix of weight of layer i from a dict
        W = model.params[f"W{i}"]
        # in FullyConnectedNet.__init__ (PART 4) we have --> self.reg = reg (Step 2 --> reg = 1e-2)
        reg_loss += 0.5 * model.reg * np.sum(W * W)
    print("Regularization loss:", reg_loss)

    # Step 5.5 : Total Loss + dout ready -> Backward Pass
    total_loss = data_loss + reg_loss
    print("Total loss:", total_loss)

    # dout — the gradient of the loss w.r.t. scores, ready to hand to Step 6.
    # For softmax + cross-entropy this has a clean closed form:
    # probs, minus 1 at the correct class, averaged over the batch.
    dout = probs.copy()
    # choose the value of each row(image) from correct class(y)
    dout[np.arange(N), y] -= 1
    dout /= N
    print("dout shape:", dout.shape)   # same shape as scores -> (N, C)

    # -------------------------------------------------------------------------
    # Step 6 : Backward Pass & Gradient
    # -------------------------------------------------------------------------
    grads = {}

    # Step 6.1 : Receive dout from Softmax
    # dout computed at the end of Step 5

    # Step 6.2 : Final Affine Backward
    # dout : how much wrong the out is  # dW_last : how much W needs change  # db_last : how much b needs change
    dout, dW_last, db_last = affine_backward(dout, final_cache)

    # Aim : how much W and B in the last layer needs to change
    dW_last += model.reg * model.params[f"W{model.num_layers}"]

    grads[f"W{model.num_layers}"] = dW_last
    grads[f"b{model.num_layers}"] = db_last
    print(f"  W{model.num_layers} grad shape:", dW_last.shape)

    # Step 6.3 : Backward through Hidden Layers -- dropout -> relu -> batchnorm -> affine,
    # repeats in reverse for each hidden layer
    # PRESENTATION : This loop applies the chain rule backward through each hidden layer
    # (dropout → ReLU → batchnorm → affine) to compute gradients for W, b, gamma, and beta.
    for i in range(model.num_layers - 1, 0, -1):
        dout = dropout_backward(dout, dropout_caches[i - 1])
        dout = relu_backward(dout, relu_caches[i - 1])

        dout, dgamma, dbeta = batchnorm_backward_alt(dout, norm_caches[i - 1])
        grads[f"gamma{i}"] = dgamma
        grads[f"beta{i}"] = dbeta

        dout, dW, db = affine_backward(dout, affine_caches[i - 1])
        dW += model.reg * model.params[f"W{i}"]
        grads[f"W{i}"] = dW
        grads[f"b{i}"] = db
        print(f"  W{i} grad shape:", dW.shape, f" gamma{i} grad shape:", dgamma.shape)

    # Step 6.4 : All Gradients Ready -> return loss, grads
    print("Gradient keys:", sorted(grads.keys()))

    # -------------------------------------------------------------------------
    # Step 7 : Optimizer & Parameter Update
    # -------------------------------------------------------------------------

    # Step 7.1 : Receive Current Parameters (w, dw, config)
    update_rule = adam            # swap for sgd / sgd_momentum / rmsprop
    learning_rate = 1e-3

    # optim_configs holds one config dict per parameter, so each parameter
    # (W1, b1, W2, b2, ...) tracks its own momentum/cache/m/v state independently.
    if "optim_configs" not in dir():
        optim_configs = {}
        for p in model.params:
            optim_configs[p] = {"learning_rate": learning_rate}

    # Step 7.2 : Choose Update Rule  <- SGD / Momentum / RMSProp / Adam
    # already chosen above via `update_rule` — swap the function reference to change it

    # Step 7.3 : Compute Next Parameters
    next_params = {}
    for p, w in model.params.items():
        dw = grads[p]
        config = optim_configs[p]
        next_w, next_config = update_rule(w, dw, config)
        next_params[p] = next_w
        optim_configs[p] = next_config
        print(f"  {p} updated, config keys: {list(next_config.keys())}")

    # Step 7.4 : Update Model Parameters -> loop back to Step 3.4 for next mini-batch
    model.params = next_params
    print("Parameters updated for this iteration.")
