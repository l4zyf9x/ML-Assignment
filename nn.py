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

    def backward(self):
        pass


class Dense(Layer):
    """ Implement fully connected dense layer.

    Recieve input [batch_size,..., d_n] and produce output [batch_size,..., num_units]
    folow expression: y = activation(x.w + b)

    Args:
      input_shape : Shape of input [batch_size,..., d_n]
      num_units : length of output chanel
      is_training : Determine whether is trainging or not
      activation: Default is not using activation
    """

    def __init__(self, input_shape, num_units, activation=None):
        super(Dense, self).__init__()
        if(len(input_shape) < 1):
            raise ValueError('Input shape has least one batch size dimension')
        self.cache = {}
        self.grads = {}
        self.input_shape = input_shape
        self.output_shape = input_shape.copy()
        self.output_shape[-1] = num_units
        self.has_weights = True
        # Caculate shape
        # input shape => [batch_size, d1,..., d_n-1, d_n]
        # output shape => [batch_size, d1,..., d_n-1, num_units]
        # weight shape => [d_n, num_units]
        # bias shape => [num_units]
        self.bias_shape = [num_units]
        self.weight_shape = [input_shape[-1], num_units]

        self.parameters['W'] = self.initiate_vars(self.weight_shape)
        self.parameters['b'] = self.initiate_vars_zero(self.bias_shape)

    def initiate_vars(self, shape, distribution=None):
        if distribution != None:
            raise ValueError(
                'Not implement distribution for initiating variable')
        return np.random.normal(size=shape)

    def initiate_vars_zero(self, shape):
        return np.zeros(shape)

    def forward(self, x, is_training=True):
        # check whether data has valid shape or not
        if is_training:
            self.cache['x'] = x
        recieved_shape = list(x.shape)
        if(len(recieved_shape) != len(self.input_shape)):
            raise ValueError('Data has shape {} is not compatible with {}'.format(
                x.shape, self.input_shape))
        recieved_shape[self.input_shape == -1] = -1
        if(recieved_shape != self.input_shape):
            raise ValueError('Data has shape {} is not compatible with {}'.format(
                x.shape, self.input_shape))
        output = np.matmul(
            x, self.parameters['W']) + self.parameters['b']
        return output

    def backward(self, dy):
        # expected dy => [batch_size, ..., d_n-1, num_units]
        recieved_shape = list(dy.shape)
        if(len(recieved_shape) != len(self.output_shape)):
            raise ValueError('d_y has shape {} is not compatible with {}'.format(
                dy.shape, self.output_shape))
        recieved_shape[self.input_shape == -1] = -1
        if(recieved_shape != self.output_shape):
            raise ValueError('d_y has shape {} is not compatible with {}'.format(
                dy.shape, self.output_shape))
        batch_size = dy.shape[0]
        self.grads['db'] = np.sum(dy, axis=0) / batch_size
        if len(self.input_shape) == 2:
            self.grads['dW'] = np.sum(np.matmul(np.expand_dims(
                self.cache['x'], 2), np.expand_dims(dy, axis=1)), axis=0) / batch_size
        else:
            self.grads['dW'] = np.sum(np.matmul(np.swapaxes(
                self.cache['x'], -1, -2), dy), axis=0, keepdims=True) / batch_size
        dx = np.matmul(dy, np.swapaxes(self.parameters['W'], -1, -2))

        return dx

    def apply_grads(self, learning_rate=0.001, l2_penalty=1e-4):
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

    def forward(self, x, is_training=True):
        if(is_training):
            self.cache['x'] = x
        y = x.copy()
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

    def forward(self, data, is_training=True):
        if(len(data.shape) != 2):
            raise ValueError(
                'data have shape is not compatible. Expect [batch_size, nums_score]')
        logits = np.exp(data - np.amax(data, axis=1, keepdims=True))
        logits = logits / np.sum(logits, axis=1, keepdims=True)
        if is_training:
            self.cache['logits'] = np.copy(logits)

        # print(logits)
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


class CELoss(Layer):
    """ Cross Entropy Loss

    Loss function: L = sum(y.log(s)) / batch_size ,
        y is labels,
        s is predict
    Derivative of Loss: dL/ds_i= y_i/s_i
    """

    def __init__(self):
        self.cache = {}
        self.has_weights = False
        super(CELoss, self).__init__()

    def forward(self, logits, labels, is_training=True):
        logits[logits < 1e-10] = 1e-8
        # logits => [batch_size, num_units]
        # labels => [batch_size, num_units]
        if is_training:
            self.cache['labels'] = labels
            self.cache['logits'] = logits
        batch_size = logits.shape[0]
        loss = - np.sum(labels * np.log(logits),
                      keepdims=True) / batch_size
        return loss

    def backward(self, loss):
        # => [batch_size, num_units]

        return - self.cache['labels'] / (self.cache['logits'] * self.cache['logits'].shape[0])


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

    def train(self, data, labels, batch_size=256, epochs=50, display_after=50):
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
                    batch_preds = layer.forward(batch_preds)
                loss = self.loss.forward(batch_preds, y_batch)
                dA = self.loss.backward(loss)
                for layer in reversed(self.model):
                    dA = layer.backward(dA)

                for layer in self.model:
                    if layer.has_weights:
                        layer.apply_grads()
                iter += 1
                if iter % display_after == 0:
                    print('Epoch {} patch {}, loss: {}'.format(
                        epoch, iter % display_after, loss))

    def predict(self, data):

        predictions = np.zeros(data.shape[0], self.num_classes)
        batch_preds = data.copy()
        for layer in self.model:
            predictions = layer.forward(batch_preds)
        return predictions

    def evaluate(self, data, labels):
        predictions = self.predict(data)
        if(predictions.shape != labels.shape):
            raise ValueError('prediction shape does not match labels shape')
        return np.mean(np.argmax(labels, axis=0) == np.argmax(predictions, axis=0))


# num_classes = 10
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# fashion_mnist = keras.datasets.fashion_mnist
# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# train_images = train_images / 255.0
# test_images = test_images / 255.0
# train_images = train_images.reshape((-1, 784))
# test_images = test_images.reshape((-1, 784))

# labels = np.zeros((train_labels.shape[0], 10))
# labels[np.arange(train_labels.shape[0]), train_labels] = 1
# train_labels = labels
# labels = np.zeros((test_labels.shape[0], 10))
# labels[np.arange(test_labels.shape[0]), test_labels] = 1
# test_labels = labels

# model = Model(Dense(input_shape=[-1, 784], num_units=300),
#               Relu(),
#               Dense(input_shape=[-1, 300], num_units=300),
#               Relu(),
#               Dense(input_shape=[-1, 300], num_units=10),
#               Softmax())
# model.set_loss(CELoss())

# model.train(train_images, train_labels)
