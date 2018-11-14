from neural import *
from numerical_grads import *


class TestClass(object):
    def test_dense_forward(self):
        input = np.ones(shape=[10, 5])
        dense = Linear(num_in=5, num_out=20)
        output = dense.forward(input)

        assert list(dense.parameters['W'].shape) == [5, 20]
        assert list(dense.parameters['b'].shape) == [20]
        assert list(output.shape) == [10, 20]

    def test_dense_backward(self):
        input = np.arange(2*3).reshape((2, 3))
        dense = Linear(num_in=3, num_out=4)

        dense.forward(input, is_training=True)
        dy = np.ones((2, 4))
        dense.backward(dy)
        grads = dense.grads

        numgrads = compute_num_grads_1(dense, input)
        norms = np.array([(np.linalg.norm(grads[idx]-numgrads[idx]) /
                           np.linalg.norm(grads[idx]+numgrads[idx])) for idx in grads])

        assert list(grads['db'].shape) == [4]
        assert list(grads['dW'].shape) == [3, 4]
        assert np.all(norms < 1e-6)

    def test_dense_backward_2(self):
        input = np.arange(20*40, dtype=float).reshape((20, 40))
        dense = Linear(num_in=40, num_out=60)

        dense.forward(input, is_training=True)
        dy = np.ones((20, 60))
        dx = dense.backward(dy)
        dx_num = compute_num_grads_2(dense, input)

        norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
        assert norm < 1e-6

    def test_forward_softmax(self):
        input = np.arange(4*5).reshape((4, 5))
        softmax = Softmax()
        output = softmax.forward(input)
        assert list(output.shape) == [4, 5]

    def test_relu(self):
        relu = Relu()
        input = np.array([[2, -2, 3],
                          [-1, 2, 3]], dtype=float)
        output = relu.forward(input)
        dy = np.ones(shape=(2, 3))
        dx = relu.backward(dy)
        dx_num = compute_num_grads_2(relu, input)
        norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
        assert np.all(output == np.array([[2, 0, 3],
                                          [0, 2, 3]], dtype=float))
        assert list(output.shape) == [2, 3]
        assert norm < 1e-6

    def test_cross_entropy(self):
        ce_loss = CELoss()
        input = np.array([[3, 4, 6, 10],
                          [4, 2, 1, 3],
                          [2, 5, 8, 9]], dtype=float)
        label = np.array([[1, 4, 4, 17],
                          [5, 3, 1, 2],
                          [3, 9, 6, 7]], dtype=float)

        loss = ce_loss.compute_loss(input, label)
        dx = ce_loss.compute_derivation(input, label)
        dx_num = compute_num_grads_3(ce_loss, input, label)
        norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
        assert norm < 1e-6

    def test_softmax_celoss(self):
        softmax = Softmax()
        ce_loss = CELoss()
        input = np.array([[3, 4, 6, 10],
                          [4, 2, 1, 3],
                          [2, 5, 8, 9]], dtype=float)
        label = np.array([[1, 4, 4, 17],
                          [5, 3, 1, 2],
                          [3, 9, 6, 7]], dtype=float)
        o1 = softmax.forward(input)
        loss = ce_loss.compute_loss(o1, label)
        do1 = ce_loss.compute_derivation(o1, label)
        dx = softmax.backward(do1)

        dx_num = compute_num_grads_4(softmax, ce_loss, input, label)
        norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
        assert norm < 1e-6

    # def test_backward_softmax(self):
    #     input = np.arange(2*3, dtype=float).reshape((2, 3)) + 2
    #     softmax = Softmax()
    #     output = softmax.forward(input)
    #     print(output)
    #     dy = np.ones(shape=(2, 3))
    #     dx = softmax.backward(dy)
    #     dx_num = compute_num_grads_2(softmax, input.copy())
    #     # print(dx_num)
    #     norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
    #     print('dx : \n', norm)
    #     assert list(output.shape) == [2, 3]
    #     assert norm < 1e-6
