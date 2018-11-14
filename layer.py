import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


class Layer:
    """ Base class for neural network.
        Two abstract funtion forward and backward to help inference and
        backprogation
    """

    def __init__(self):
        self.parameters = {}
        self.has_weights = False

    def forward(self, x, is_training=True):
        pass

    def backward(self, dy):
        pass


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=(1, 1)):
    """ An implementation of im2col based on some fancy indexing 

    Args: 
        x_shape : [B, Cin, H, W]
        field_height: kH
        field_width: kW

    Return: [B*OH*OW, kH*kW*Cin]
    """

    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride[0] == 0
    assert (W + 2 * padding - field_height) % stride[1] == 0
    out_height = (H + 2 * padding - field_height) // stride[0] + 1
    out_width = (W + 2 * padding - field_width) // stride[1] + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, filter=(2, 2), padding=1, stride=(1, 1)):
    """ An implementation of im2col based on some fancy indexing 

    Args: 
        x : [B, H, W, Cin]
        field_height: kH
        field_width: kW

    Return: [B*OH*OW, kH*kW*Cin]
    """

    # x => [B, Cin, H, W]
    x = x.tranpose(0, 3, 1, 2)

    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, filter[0], filter[1], padding,
                                 stride)
    # cols => [B, kH*kW*Cin, OH*OW]
    cols = x_padded[:, k, i, j]
    C = x.shape[1]

    # cols => [B*OH*OW, kH*kW*Cin]
    cols = cols.transpose(0, 2, 1).reshape(-1, filter[0] * filter[1] * C)
    return cols


def col2im_indices(cols, x_shape, filter=(2, 2), padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at 

    Args:
        cols: [B*OH*OW, kH*kW*Cin]
        x_shape: Shape of initial input (B, H, W, Cin)
        filter: Shape of filter (kH, kW)

    Returns: [B, H, W, Cin]


    """

    N, H, W, C = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, filter[0], filter[1], padding,
                                 stride)
    # cols => [B*OH*OW, kH*kW*Cin]
    # cols_reshaped => [B, OH*OW, kH*kW*Cin]
    cols_reshaped = cols.reshape(N, -1, C * filter[0] * filter[1])

    # [B, C * kH * kW, OH*OW]
    cols_reshaped = cols_reshaped.transpose(0, 2, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding].tranpose(0, 2, 3, 1)


class Conv2d(Layer):
    """ An implementation of Convolution 2D"""
    def __init__(self, input_shape=(-1, 3, 3, 1), filter=(1, 2, 2, 1), stride=(1, 1), padding='VALID'):
        super(Conv2d, self).__init__()
        self.has_weights = True
        self.input_shape = input_shape
        self.filter = filter
        self.stride = stride
        self.cache = {}
        self.w_shape = []
        self.padding = 0
        self.out_height = (input_shape[1] + 2 * padding -
                           filter[1]) // stride[0] + 1
        self.out_width = (input_shape[2] + 2 * padding -
                          filter[2]) // stride[1] + 1

        self.linear_in = filter[1] * filter[2] * filter[3]
        self.linear_out = filter[0]
        self._linear = Linear(self.linear_in, self.linear_out)

    def forward(self, x, is_training=True):
        """Implement forward for Conv2d

        x: [B, H, W, Cin]
        """
        B, _, _, _ = x.shape
        self.input_shape = x.shape

        # x2row =>  [B*H*W, Cin]
        x2row = im2col_indices(
            x, filter=(self.filter[1], self.filter[2]), padding=self.padding, stride=self.stride)
        assert x2row.shape == [B * self.out_height *
                               self.out_height, self.linear_in]

        # out [B*OH*OW, Cout]
        out = self._linear.forward(x2row)
        # out [B, OH, OW, Cout]
        return out.reshape((B, self.out_height, self.out_width, self.linear_out))

    def backward(self, dy):
        """Implement forward for Conv2d

        dy: [B, H, W, Cout]
        """
        # dy => [B, OH, OW, Cout]
        dy = dy.reshape(-1, self.linear_out)
        # dx => [B]
        dx = self._linear.backward(dy)

        # cols => [B* OH*OW, kH*kW*Cin] => [kH*kW*Cin ,OH*OW, B] => [kH*kW*Cin ,OH*OW*B]
        # dx = dx.reshape(-1, self.out_height*self.out_width,
        #                 self.linear_in).tranpose(2, 1, 0).reshape(self.linear_in, -1)
        #
        dx = col2im_indices(cols=dx, x_shape=self.input_shape, filter=(
            self.filter[1], self.filter[2]), padding=self.padding)

        return dx

    def apply_grads(self, learning_rate=0.01, l2_penalty=1e-4):
        self._linear.apply_grads(learning_rate=0.01, l2_penalty=1e-4)


class Flatten(Layer):
    def __init__(self, input_shape):
        super(Flatten, self).__init__()
        self.input_shape = input_shape
        self.out_num = np.prod(input_shape[1:])

    def forward(self, x):
        return x.reshape(-1, self.out_num)

    def backward(self, dy):
        return dy.reshape(tuple(self.input_shape))


class Linear(Layer):
    """ Implement fully connected dense layer.

    Recieve input [batch_size, num_in] and produce output [batch_size, num_out]
    folow expression: y = activation(x.w + b)

    Args:
      input_shape: (int) length input
      num_units: (int) length of output 
      is_training: (bool) Determine whether is trainging or not
      activation: Default is not using activation
    """

    def __init__(self, num_in, num_out, activation=None):
        super(Linear, self).__init__()
        self.cache = {}
        self.grads = {}
        self.bias_shape = [num_out]
        self.weight_shape = [num_in, num_out]
        self.has_weights = True
        self.parameters['W'] = self.initiate_vars(self.weight_shape)
        self.parameters['b'] = self.initiate_vars_zero(self.bias_shape)
        self.name = 'Linear'

    def initiate_vars(self, shape, distribution=None):
        if distribution != None:
            raise ValueError(
                'Not implement distribution for initiating variable')

        # print(w)
        return np.random.randn(shape[0], shape[1]) * 1e-4

    def initiate_vars_zero(self, shape):
        return np.zeros(shape)

    def forward(self, x, is_training=True):
        # check whether data has valid shape or not
        if is_training:
            self.cache['x'] = x.copy()
            self.batch_size = x.shape[0]
        y = np.dot(x, self.parameters['W']) + self.parameters['b']

        return y

    def backward(self, dy):
        self.grads['db'] = np.sum(dy, axis=0)
        self.grads['dW'] = np.dot(self.cache['x'].T, dy)
        dx = np.dot(dy, self.parameters['W'].T)

        return dx

    def apply_grads(self, learning_rate=0.01, l2_penalty=1e-4):
        self.parameters['W'] -= learning_rate * \
            (self.grads['dW'] + l2_penalty * self.parameters['W'])
        self.parameters['b'] -= learning_rate * \
            (self.grads['db'] + l2_penalty * self.parameters['b'])


class Relu(Layer):
    """ Implement Relu activation function:
    y = x with x >= 0
    y = 0 with x<0
    """

    def __init__(self):

        super(Relu, self).__init__()
        self.cache = {}
        self.has_weights = False
        self.name = 'Relu'

    def forward(self, x, is_training=True):
        if(is_training):
            self.cache['x'] = x.copy()
        y = x
        y[y < 0] = 0
        return y

    def backward(self, dy):
        dy[self.cache['x'] <= 0] = 0
        return dy


class Softmax(Layer):
    """ Implement Softmax activation function

    Recieve input with shape [batch_size, num_score] and produce output folowing
    expression:  y = e^score / sum(e^score)

    Args:
    """

    def __init__(self):
        super(Softmax, self).__init__()
        self.cache = {}
        self.has_weights = False
        self.name = 'Softmax'

    def forward(self, data, is_training=True):
        # print(data[0])
        if(len(data.shape) != 2):
            raise ValueError(
                'data have shape is not compatible. Expect [batch_size, nums_score]')
        logits = np.exp(data - np.amax(data, axis=1, keepdims=True))
        logits = logits / np.sum(logits, axis=1, keepdims=True)
        if is_training:
            self.cache['logits'] = np.copy(logits)

        # print(logits[0])
        return logits

    def backward(self, dy):
        if(len(dy.shape) != 2):
            raise ValueError(
                'data have shape is not compatible. Expect [batch_size, nums_score]')
        num_units = dy.shape[-1]

        # [batch_size, num_units, 1] . [batch_size, 1, num_units]
        # = [batch_size, num_units, num_units]
        # Represent of matrix ds:
        #   [ds1/dx1 ds1/dx2 ... ds1/dxN]
        #   [ds2/dx1 ds2/dx2 ... ds2/dxN]
        #   [  ...           ...    ... ]
        #   [dsN/dx1 dsN/dx2 ... dsN/dxN]
        # with ds_i/dx_j = S_i(1 - S_j)  , with i==j
        #                = -S_i*S_j      , with i!=j
        ds = -np.matmul(np.expand_dims(self.cache['logits'], axis=-1),
                        np.expand_dims(self.cache['logits'], axis=1))
        ds[:, np.arange(num_units), np.arange(
            num_units)] += self.cache['logits']

        dx = np.matmul(np.expand_dims(dy, axis=1), ds)
        return np.squeeze(dx, axis=1)
        # return dy


class CELoss():
    """ Cross Entropy Loss

    Loss function: L = sum(y.log(s)) / batch_size ,
        y is labels,
        s is predict
    Derivative of Loss: dL/ds_i= y_i/s_i
    """

    def __init__(self):
        self.cache = {}
        self.has_weights = False
        self.eps = 1e-8
        # super(CELoss, self).__init__()

    def compute_loss(self, logits, labels, is_training=True):
        # logits = np.clip(logits, self.eps, 1. - self.eps)
        # logits => [batch_size, num_units]
        # labels => [batch_size, num_units]
        if is_training:
            self.cache['labels'] = labels
            self.cache['logits'] = logits
        self.batch_size = logits.shape[0]
        loss = - np.sum(labels * np.log(logits)) / self.batch_size
        return loss

    def compute_derivation(self, logits, labels):
        # => [batch_size, num_units]

        # return (logits - labels)/self.batch_size
        # return - self.cache['labels'] / (self.cache['logits'] * self.cache['logits'].shape[0])
        return - self.cache['labels'] / (self.cache['logits'] * self.batch_size)


class Model:
    def __init__(self, *model, **kwargs):
        self.model = model
        self.num_classes = 0
        self.batch_size = 0
        self.loss = None
        self.optimizer = None
        self.name = kwargs['name'] if 'name' in kwargs else None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_loss(self, loss):
        self.loss = loss

    # def load_weights(self):
    #     for layer in self.model:
    #         if layer.has_weights():
    #             layer.load_weights(path.join(get_models_path(), self.name))
    def get_batches(self, data, labels, batch_size=256, shuffle=True):
        N = data.shape[0]
        num_batches = N // batch_size
        if(shuffle):
            rand_idx = np.random.permutation(data.shape[0])
            data = data[rand_idx]
            labels = labels[rand_idx]
        for i in np.arange(num_batches):
            yield (data[i*batch_size:(i+1) * batch_size], labels[i*batch_size:(i+1) * batch_size])
        if N % batch_size != 0 and num_batches != 0:
            yield (data[batch_size*num_batches:], labels[batch_size*num_batches:])

    def train(self, data, labels,
              batch_size=1024, epochs=50,
              display_after=50, learning_rate=0.01,
              l2_penalty=1e-4,
              learning_rate_decay=0.95):
        if self.loss is None:
            raise RuntimeError("Set loss first using 'model.set_loss(<loss>)'")
        self.set_batch_size(batch_size)
        self.set_num_classes(labels.shape[1])

        iter = 0
        for epoch in range(epochs):
            print('Running Epoch:', epoch + 1)

            for i, (x_batch, y_batch) in enumerate(self.get_batches(data, labels)):
                batch_preds = x_batch.copy()
                for num, layer in enumerate(self.model):
                    batch_preds = layer.forward(batch_preds, is_training=True)

                loss = self.loss.compute_loss(
                    logits=batch_preds, labels=y_batch)
                dA = self.loss.compute_derivation(
                    logits=batch_preds, labels=y_batch)

                for layer in reversed(self.model):
                    dA = layer.backward(dA)

                for layer in self.model:
                    if layer.has_weights:
                        layer.apply_grads(
                            learning_rate=learning_rate, l2_penalty=l2_penalty)
                iter += 1
                if iter % display_after == 0:
                    train_acc = self.evaluate(x_batch, y_batch)
                    print('Step {}, loss: {}, train_acc: {}'.format(
                        iter, loss, train_acc))
                # break
            learning_rate *= learning_rate_decay

    def predict(self, data):
        batch_preds = data.copy()
        for layer in self.model:
            batch_preds = layer.forward(batch_preds)
        return batch_preds

    def evaluate(self, data, labels):
        predictions = self.predict(data)
        if(predictions.shape != labels.shape):
            raise ValueError('prediction shape does not match labels shape')
        return np.mean(np.argmax(labels, axis=1) == np.argmax(predictions, axis=1))

class Conv2D(Layer):

    def __init__(self, num_filter, kernal_size, activation=None, stride=(1,1), pading=None, input_shape=None):
        super().__init__()
        self.cache = {}
        self.grads = {}
        self.num_filter = num_filter
        self.kernal_size = kernal_size
        self.stride = stride
        self.pading = pading
        self.activation = activation
        self.input_shape = input_shape


    def forward(self, x, is_training=True):
        
        N, H, W, C  = x.shape
        
        w_shape = (self.num_filter, self.kernal_size[0], self.kernal_size[1], C)
        w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
         
        out_height = 1 + (H + 2*self.pading - self.kernal_size[0])//self.stride[0]
        out_width = 1 + (W + 2*self.pading - self.kernal_size[1])//self.stride[1]
       
        pad_width = ((0,0), (self.pading,self.pading), (self.pading,self.pading), (0,0))
        x = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0) 
        out = np.zeros((N, out_height, out_width, self.num_filter))

        for i in range(N):
            for z in range(self.num_filter):
                for j in range(out_height):
                    for k in range(out_width):
                        out[i, j, k, z] = np.sum(
                            x[i, j*self.stride[0]:(j*self.stride[0]+self.kernal_size[0]), k*self.stride[1]:(k*self.stride[1]+self.kernal_size[1]),:]*w[z, :, :, :])

        if is_training:
            self.cache['x'] = x
            self.cache['w'] = w
  
        return out

    def backward(self, dy):


        pass

num_classes = 10
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# train_images = train_images / 255.0
# test_images = test_images / 255.0
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

labels = np.zeros((train_labels.shape[0], 10))
labels[np.arange(train_labels.shape[0]), train_labels] = 1
train_labels = labels
labels = np.zeros((test_labels.shape[0], 10))
labels[np.arange(test_labels.shape[0]), test_labels] = 1
test_labels = labels

model = Model(Linear(num_in=784, num_out=10),
              Relu(),
              Linear(num_in=10, num_out=10),
              Softmax())
model.set_loss(CELoss())

model.train(train_images, train_labels, learning_rate=0.0001, l2_penalty=0.)
