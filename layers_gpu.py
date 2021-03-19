import cupy as np


def activation_function(act_func):
    if act_func == "relu":
        return Relu()
    if act_func == "sigmoid":
        return Sigmoid()
    if act_func == "tanh":
        return Tanh()
    if act_func == "mish":
        return Mish()
    if act_func == "tanhexp":
        return TanhExp()
    else:
        print("{} is not defined.".format(act_func))
        return None


def loss_function(loss_func):
    if loss_func == "MSE_RE":
        return MSE_RelativeError()
    if loss_func == "MSE_AE":
        return MSE_AbsoluteError()
    else:
        print("{} is not defined.".format(loss_func))
        return None


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x, is_training):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x, is_training):
        self.mask = (x <= 0)
        x[self.mask] = 0
        return x

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x, is_training):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class Tanh:
    def __init__(self):
        self.out = None

    def forward(self, x, is_training):
        self.out = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out**2)


class Mish:
    def __init__(self):
        self.x = None
        self.expx = None

    def forward(self, x, is_training):
        self.x = x
        self.expx = np.exp(x)
        return self.x * np.tanh(np.log(1.0 + self.expx))

    def backward(self, dout):
        diff = 4.0 * (self.x + 1.0 + self.expx**2) + self.expx**3 + (4.0 * self.x + 6.0) * self.expx
        diff *= self.expx
        diff /= (2.0 * self.expx + self.expx**2 + 2.0)**2
        return dout * diff


class TanhExp:
    def __init__(self):
        self.x = None

    def forward(self, x, is_training):
        self.x = x
        return x * np.tanh(np.exp(x))

    def backward(self, dout):
        return dout * (np.tanh(np.exp(self.x)) - self.x * np.exp(self.x) * (np.tanh(np.exp(self.x))**2 - 1.0))


class Identity:
    def forward(self, x, is_training):
        return x

    def backward( self, dout ):
        return dout


class MSE_RelativeError:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y, self.t = y, t
        error = (y - t) / t
        return np.mean(error**2)

    def backward(self, dout = 1.0):
        return dout * (2.0 * (self.y - self.t) / self.t**2) / float(self.y.size)


class MSE_AbsoluteError:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, y, t):
        self.y = y
        self.t = t
        error = (y - t)
        return np.mean(error**2)

    def backward(self, dout = 1):
        return dout * 2.0 * (self.y - self.t) / float(self.y.size)


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, is_training):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, is_training)

        return out.reshape(*self.input_shape)

    def __forward(self, x, is_training):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if is_training:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
