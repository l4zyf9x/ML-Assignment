from layer import *
from numerical_grads import *


class TestClass(object):
    # def test_dense_forward(self):
    #     input = np.ones(shape=[10, 5])
    #     dense = Linear(num_in=5, num_out=20)
    #     output = dense.forward(input)

    #     assert list(dense.parameters['W'].shape) == [5, 20]
    #     assert list(dense.parameters['b'].shape) == [20]
    #     assert list(output.shape) == [10, 20]

    # def test_dense_backward(self):
    #     input = np.arange(2*3).reshape((2, 3))
    #     dense = Linear(num_in=3, num_out=4)

    #     dense.forward(input, is_training=True)
    #     dy = np.ones((2, 4))
    #     dense.backward(dy)
    #     grads = dense.grads

    #     numgrads = compute_num_grads_1(dense, input)
    #     norms = np.array([(np.linalg.norm(grads[idx]-numgrads[idx]) /
    #                        np.linalg.norm(grads[idx]+numgrads[idx])) for idx in grads])

    #     assert list(grads['db'].shape) == [4]
    #     assert list(grads['dW'].shape) == [3, 4]
    #     assert np.all(norms < 1e-6)

    # def test_dense_backward_2(self):
    #     input = np.arange(20*40, dtype=float).reshape((20, 40))
    #     dense = Linear(num_in=40, num_out=60)

    #     dense.forward(input, is_training=True)
    #     dy = np.ones((20, 60))
    #     dx = dense.backward(dy)
    #     dx_num = compute_num_grads_2(dense, input)

    #     norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
    #     assert norm < 1e-6

    # def test_forward_softmax(self):
    #     input = np.arange(4*5).reshape((4, 5))
    #     softmax = Softmax()
    #     output = softmax.forward(input)
    #     assert list(output.shape) == [4, 5]

    # def test_relu(self):
    #     relu = Relu()
    #     input = np.array([[2, -2, 3],
    #                       [-1, 2, 3]], dtype=float)
    #     output = relu.forward(input)
    #     dy = np.ones(shape=(2, 3))
    #     dx = relu.backward(dy)
    #     dx_num = compute_num_grads_2(relu, input)
    #     norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
    #     assert np.all(output == np.array([[2, 0, 3],
    #                                       [0, 2, 3]], dtype=float))
    #     assert list(output.shape) == [2, 3]
    #     assert norm < 1e-6

    # def test_cross_entropy(self):
    #     ce_loss = CELoss()
    #     input = np.array([[3, 4, 6, 10],
    #                       [4, 2, 1, 3],
    #                       [2, 5, 8, 9]], dtype=float)
    #     label = np.array([[1, 4, 4, 17],
    #                       [5, 3, 1, 2],
    #                       [3, 9, 6, 7]], dtype=float)

    #     loss = ce_loss.compute_loss(input, label)
    #     dx = ce_loss.compute_derivation(input, label)
    #     dx_num = compute_num_grads_3(ce_loss, input, label)
    #     norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
    #     assert norm < 1e-6

    # def test_softmax_celoss(self):
    #     softmax = Softmax()
    #     ce_loss = CELoss()
    #     input = np.array([[3, 4, 6, 10],
    #                       [4, 2, 1, 3],
    #                       [2, 5, 8, 9]], dtype=float)
    #     label = np.array([[1, 4, 4, 17],
    #                       [5, 3, 1, 2],
    #                       [3, 9, 6, 7]], dtype=float)
    #     o1 = softmax.forward(input)
    #     loss = ce_loss.compute_loss(o1, label)
    #     do1 = ce_loss.compute_derivation(o1, label)
    #     dx = softmax.backward(do1)

    #     dx_num = compute_num_grads_4(softmax, ce_loss, input, label)
    #     norm = np.linalg.norm(dx - dx_num) / np.linalg.norm(dx + dx_num)
    #     assert norm < 1e-6

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

    def test_convolution(self):
        np.random.seed(0)
        
        x = np.random.randint(1, 10, size=(3,4,4,1)).astype(float)
        y = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0]]).astype(float)

        eps = 1e-7
        model = Model(Conv2d(input_shape=(-1, 4, 4, 1), filter=(5, 2, 2, 1)),
              Flatten(input_shape= (-1, 3, 3, 5)),
              Linear(num_in=3*3*5, num_out=5),
              Softmax())
        model.set_loss(CELoss())
        para_cp = model.model[0]._linear.parameters['W'].copy()
        flat = model.model[0]._linear.parameters['W'].reshape(-1)
        num_grads = np.zeros(flat.shape)

        batch_preds = x.copy()
        for  layer in model.model:
            batch_preds = layer.forward(batch_preds, is_training=True)
        loss = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
        dA = model.loss.compute_derivation(
                    logits=batch_preds, labels=y)
        for layer in reversed(model.model):
            dA = layer.backward(dA)
        grads = model.model[0]._linear.grads['dW'].copy().reshape(-1)
        

        model.model[0]._linear.parameters['W'] = para_cp.copy()
        for idx,_ in enumerate(np.arange(len(flat))):
            model.model[0]._linear.parameters['W'].reshape(-1)[idx] -= eps
            batch_preds = x.copy()
            for layer in model.model:
                batch_preds = layer.forward(batch_preds, is_training=False)
            loss1 = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
            
            model.model[0]._linear.parameters['W'].reshape(-1)[idx] += 2*eps
            batch_preds = x.copy()
            for  layer in model.model:
                batch_preds = layer.forward(batch_preds, is_training=False)
            loss2 = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
            num_grads[idx]= (loss2-loss1)/(2*eps)
        


        print (grads)
        print(num_grads)
        norm = np.linalg.norm(num_grads - grads) / np.linalg.norm(num_grads + grads)
        assert norm < 1e-6
        # assert False

    def test_convolution_navie(self):
        np.random.seed(0)
        
        x = np.random.randint(1, 10, size=(3,4,4,1)).astype(float)
        y = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0]]).astype(float)

        eps = 1e-7
        model = Model(Conv2DNaive(input_shape=(-1, 4, 4, 1), filter=(5, 2, 2, 1)),
              Flatten(input_shape= (-1, 3, 3, 5)),
              Linear(num_in=3*3*5, num_out=5),
              Softmax())
        model.set_loss(CELoss())
        para_cp = model.model[0].parameters['W'].copy()
        flat = model.model[0].parameters['W'].reshape(-1)
        num_grads = np.zeros(flat.shape)

        batch_preds = x.copy()
        for  layer in model.model:
            batch_preds = layer.forward(batch_preds, is_training=True)
        loss = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
        dA = model.loss.compute_derivation(
                    logits=batch_preds, labels=y)
        for layer in reversed(model.model):
            dA = layer.backward(dA)
        
        
        grads = model.model[0].grads['dW'].copy().reshape(-1)
        

        model.model[0].parameters['W'] = para_cp.copy()
        for idx,_ in enumerate(np.arange(len(flat))):
            model.model[0].parameters['W'].reshape(-1)[idx] -= eps
            batch_preds = x.copy()
            for layer in model.model:
                batch_preds = layer.forward(batch_preds, is_training=False)
            loss1 = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
            
            model.model[0].parameters['W'].reshape(-1)[idx] += 2*eps
            batch_preds = x.copy()
            for  layer in model.model:
                batch_preds = layer.forward(batch_preds, is_training=False)
            loss2 = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
            num_grads[idx]= (loss2-loss1)/(2*eps)
        


        print (grads)
        print(num_grads)
        norm = np.linalg.norm(num_grads - grads) / np.linalg.norm(num_grads + grads)
        assert norm < 1e-6
        # assert False

    def test_maxpooling_navie(self):
        np.random.seed(0)
        
        x = np.random.randint(1, 10, size=(3,4,4,1)).astype(float)
        y = np.array([[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0]]).astype(float)

        eps = 1e-9
        model = Model(MaxPooling2DNaive(pool_size=(2, 2)),
              Flatten(input_shape= (-1, 3, 3, 1)),
              Linear(num_in=3*3, num_out=5),
              Softmax())
        model.set_loss(CELoss())



        batch_preds = x.copy()
        for  layer in model.model:
            batch_preds = layer.forward(batch_preds, is_training=True)
        loss = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
        dA = model.loss.compute_derivation(
                    logits=batch_preds, labels=y)
        for layer in reversed(model.model):
            dA = layer.backward(dA)
    
        dA = dA.reshape(-1)
        # grads = model.model[0].grads['dW'].copy().reshape(-1)
        
        num_grads = np.zeros(x.shape).reshape(-1)
        # model.model[0].parameters['W'] = para_cp.copy()
        for idx,_ in enumerate(np.arange(len(x.reshape(-1)))):
            batch_preds = x.copy()
            batch_preds.reshape(-1)[idx] -= eps
            for layer in model.model:
                batch_preds = layer.forward(batch_preds, is_training=False)
            loss1 = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
                        
            batch_preds = x.copy()
            batch_preds.reshape(-1)[idx] += eps
            for  layer in model.model:
                batch_preds = layer.forward(batch_preds, is_training=False)
            loss2 = model.loss.compute_loss(
                    logits=batch_preds, labels=y)
            num_grads[idx]= (loss2-loss1)/(2*eps)
        


        print (dA)
        print(num_grads)
        norm = np.linalg.norm(num_grads - dA) / np.linalg.norm(num_grads + dA)
        assert norm < 1e-6
        # assert False        