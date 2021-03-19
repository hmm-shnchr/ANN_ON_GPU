from collections import OrderedDict
from layers_gpu import *
from numerical_gradient_gpu import numerical_gradient
import cupy as np

class MultiLayerNetExtend:
    def __init__(self, input_size, hidden, act_func, weight_init, batch_norm, output_size, lastlayer_identity, loss_func):
        self.input_size = input_size
        self.hidden, self.act_func, self.weight_init, self.batch_norm = hidden, act_func, weight_init, batch_norm
        self.output_size = output_size
        self.lastlayer_identity = lastlayer_identity
        self.loss_func = loss_func
        self.params = {}
        self.__init_weight(self.weight_init)
        self.layers = OrderedDict()
        for i in range(1, len(self.hidden)+2):
            self.layers["Affine"+str(i)] = Affine(self.params["Weight"+str(i)], self.params["Bias"+str(i)])
            if self.batch_norm and i != len(self.hidden)+1:
                self.layers["BatchNorm"+str(i)] = BatchNormalization(self.params["gamma"+str(i)], self.params["beta"+str(i)], 1e-7)
            if i == len(self.hidden)+1 and self.lastlayer_identity:
                self.layers["Activation"+str(i)] = Identity()
                continue
            self.layers["Activation"+str(i)] = activation_function(self.act_func)
        self.lastlayer = loss_function(self.loss_func)

    def __init_weight(self, weight_init):
        all_layers = [self.input_size] + self.hidden + [self.output_size]
        for i in range(1, len(all_layers)):
            if weight_init == "he":
                scale = np.sqrt(2.0/all_layers[i-1])
            if weight_init == "xavier":
                scale = np.sqrt(1.0/all_layers[i-1])
            self.params["Weight"+str(i)] = scale * np.random.randn(all_layers[i-1], all_layers[i])
            self.params["Bias"+str(i)] = np.zeros(all_layers[i])
            if self.batch_norm and i != len(all_layers)-1:
                self.params["gamma"+str(i)] = np.ones(all_layers[i])
                self.params["beta"+str(i)] = np.zeros(all_layers[i])

    def predict(self, x, is_training):
        for layer in self.layers.values():
            x = layer.forward(x, is_training)
        return x

    def loss(self, x, t, is_training):
        y = self.predict(x, is_training)
        return self.lastlayer.forward(y, t)

    def accuracy(self, x, t, is_training):
        y = self.predict(x, is_training)
        accuracy = np.abs((y-t)/t)
        accuracy = np.sum(accuracy)/y.size
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss_func(x, t , is_training = True)
        grads = {}
        for i in range(1, len(self.hidden)+2):
            grads["Weight"+str(i)] = numerical_gradient(loss_W, self.params["Weight"+str(i)])
            grads["Bias"+str(i)] = numerical_gradient(loss_W, self.params["Bias"+str(i)])
            if self.batch_norm and i != len(self.hidden)+1:
                grads["gamma"+str(i)] = numerical_gradient(loss_W, self.params["gamma"+str(i)])
                grads["beta"+str(i)] = numerical_gradient(loss_W, self.params["beta"+str(i)])
        return grads


    def gradient(self, x, t, is_training):
        ##Forward
        self.loss(x, t, is_training)
        ##Backward
        dout = 1
        dout = self.lastlayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        ##Update gradients
        grads = {}
        for i in range(1, len(self.hidden)+2):
            grads["Weight"+str(i)] = self.layers["Affine"+str(i)].dW
            grads["Bias"+str(i)] = self.layers["Affine"+str(i)].db
            if self.batch_norm and i != len(self.hidden)+1:
                grads["gamma"+str(i)] = self.layers["BatchNorm"+str(i)].dgamma
                grads["beta"+str(i)] = self.layers["BatchNorm"+str(i)].dbeta
        return grads
