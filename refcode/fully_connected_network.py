from layers.fully_connected import FullyConnected
from layers.flatten import Flatten
from layers.activation import Elu, Softmax

from utilities.filereader import get_data
from utilities.model import Model

from loss.losses import CategoricalCrossEntropy
from tensorflow import keras

import numpy as np
np.random.seed(0)


if __name__ == '__main__':

    num_classes = 10
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_images = train_images.reshape((-1, 784))
    test_images = test_images.reshape((-1, 784))

    # labels = np.zeros((train_labels.shape[0], 10))
    # labels[np.arange(train_labels.shape[0]), train_labels] = 1
    # train_labels = labels
    # labels = np.zeros((test_labels.shape[0], 10))
    # labels[np.arange(test_labels.shape[0]), test_labels] = 1
    # test_labels = labels


    print("Train data shape: {}, {}".format(train_images.shape, train_labels.shape))
    print("Test data shape: {}, {}".format(test_images.shape, test_labels.shape))

    model = Model(
        Flatten(),
        FullyConnected(units=200),
        Elu(),
        FullyConnected(units=200),
        Elu(),
        FullyConnected(units=10),
        Softmax(),
        name='fcn200'
    )

    model.set_loss(CategoricalCrossEntropy)
    model.train(train_images, train_labels, batch_size=128, epochs=50)

    print('Testing accuracy = {}'.format(model.evaluate(test_images, test_labels)))

