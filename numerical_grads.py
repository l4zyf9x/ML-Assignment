from layer import *
import numpy as np


def compute_num_grads_1(layer: Layer, x):
    params = layer.parameters
    paramsInitial = params.copy()
    e = 1e-4
    numgrad = {}
    batch_size = x.shape[0]
    for w_idx in params:
        w = params[w_idx].ravel()
        numgrad['d'+w_idx] = np.zeros(w.shape)
        for p_idx in np.arange(len(w)):
            w[p_idx] -= e
            # y1,y2 => [batch_size, num_units]
            y1 = layer.forward(x, is_training=False)
            w[p_idx] += 2*e
            y2 = layer.forward(x, is_training=False)
            numgrad['d'+w_idx][p_idx] = np.sum(
                np.sum(y2, axis=1) - np.sum(y1, axis=1))/(2*batch_size*e)
        numgrad['d'+w_idx] = numgrad['d'+w_idx].reshape(params[w_idx].shape)
    layer.parameters = paramsInitial

    return numgrad


def compute_num_grads_2(layer: Layer, x):
    e = 1e-4
    batch_size = x.shape[0]
    x_reshape = x.reshape((batch_size, -1))
    numgrad = np.zeros(x_reshape.shape)
    for idx in np.arange(x_reshape.shape[1]):
        # print(x)
        x_reshape[:, idx] -= e
        y1 = layer.forward(x)
        # print(y1)
        x_reshape[:, idx] += 2*e
        y2 = layer.forward(x)
        numgrad[:, idx] = (np.sum(y2, axis=1) - np.sum(y1, axis=1))/(2*e)

        # print('> compute_num_grads_2: x =\n ',x_reshape)

    numgrad = numgrad.reshape(x.shape)
    return numgrad


def compute_num_grads_3(layer, x, y):
    e = 1e-6
    batch_size = x.shape[0]
    x_reshape = x.reshape((batch_size, -1))
    numgrad = np.zeros(x_reshape.shape)
    for batch_num in np.arange(batch_size):
        for idx in np.arange(x_reshape.shape[1]):
            x_reshape[batch_num, idx] -= e
            y1 = layer.forward(x, y)
            x_reshape[batch_num, idx] += 2*e
            y2 = layer.forward(x, y)
            numgrad[batch_num, idx] = (y2 - y1)/(2*e)

        # print('> compute_num_grads_2: x =\n ',x_reshape)

    numgrad = numgrad.reshape(x.shape)
    return numgrad

def compute_num_grads_4(layer, loss, x, y):
    e = 1e-6
    batch_size = x.shape[0]
    x_reshape = x.reshape((batch_size, -1))
    numgrad = np.zeros(x_reshape.shape)
    for batch_num in np.arange(batch_size):
        for idx in np.arange(x_reshape.shape[1]):
            x_reshape[batch_num, idx] -= e
            y1 = layer.forward(x)
            l1 = loss.forward(y1,y)
            x_reshape[batch_num, idx] += 2*e
            y2 = layer.forward(x)
            l2 = loss.forward(y2,y)
            numgrad[batch_num, idx] = (l2 - l1)/(2*e)

        # print('> compute_num_grads_2: x =\n ',x_reshape)

    numgrad = numgrad.reshape(x.shape)
    return numgrad