from typing import List
import math
import numpy as np
import numpy as np
from .Init import * 

def shape(X):
    if isinstance(X, np.ndarray):
        ret = "ndarray"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return f" {X.shape} "
    if isinstance(X, int):
        return "int"
    if isinstance(X, float):
        ret = "float"
        if np.any(np.isposinf(X)):
            ret += "_posinf"
        if np.any(np.isneginf(X)):
            ret += "_neginf"
        if np.any(np.isnan(X)):
            ret += "_nan"
        return ret
    else:
        raise NotImplementedError(f"unsupported type {type(X)}")


class Node(object):
    def __init__(self, name, *params):
        self.grad = [] # 节点的梯度，self.grad[i]对应self.params[i]在反向传播时的梯度
        self.cache = [] # 节点保存的临时数据
        self.name = name # 节点的名字
        self.params = list(params) # 用于Linear节点中存储weight和bias参数使用

    def num_params(self):
        return len(self.params)

    def cal(self, X):
        '''
        计算函数值。请在其子类中完成具体实现。
        '''
        pass

    def backcal(self, grad):
        '''
        计算梯度。请在其子类中完成具体实现。
        '''
        pass

    def flush(self):
        '''
        初始化或刷新节点内部数据，包括梯度和缓存
        '''
        self.grad = []
        self.cache = []

    def forward(self, X, debug=False):
        '''
        正向传播。输入X，输出正向传播的计算结果。
        '''
        if debug:
            print(self.name, shape(X))
        ret = self.cal(X)
        if debug:
            print(shape(ret))
        return ret

    def backward(self, grad, debug=False):
        '''
        反向传播。输入grad（该grad为反向传播到该节点的梯度），输出反向传播到下一层的梯度。
        '''
        if debug:
            print(self.name, shape(grad))
        ret = self.backcal(grad)
        if debug:
            print(shape(ret))
        return ret
    
    def eval(self):
        pass

    def train(self):
        pass


class LayerNorm(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim):
        super().__init__("LayerNorm", ones((indim[0] * indim[1])), zeros(indim[0] * indim[1]))
        self.indim = indim[0] * indim[1]

    def cal(self, X):
        B, L, D = X.shape[0], X.shape[1], X.shape[2]
        X = X.reshape(B, L*D)
        X = X.copy()
        x_mean = X.mean(axis=-1, keepdims=True)
        x_std = X.std(axis=-1, keepdims=True)
        X -= x_mean
        X /= (x_std + self.EPS)
        self.cache.append((X.copy().reshape(B, L, D), x_mean, x_std))
        X *= self.params[0]
        X += self.params[1]
        return X.reshape(B, L, D)

    def backcal(self, grad):
        X, x_mean, x_std = self.cache[-1]
        B, L, D = X.shape[0], X.shape[1], X.shape[2]
        X = X.reshape(B, L*D)
        self.grad.append(np.multiply(X, grad.reshape(B, L*D)).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(B, L*D).reshape(-1, self.indim).sum(axis=0))
        return (grad.reshape(B, L*D)*np.expand_dims(self.params[0], 0) / (x_std + self.EPS)).reshape(B, L, D)


class BatchNorm(Node):
    '''
    input shape (*)
    output (*)
    '''
    EPS = 1e-8
    def __init__(self, indim, momentum: float = 0.9):
        super().__init__("batchnorm", ones((indim)), zeros(indim))
        self.momentum = momentum
        self.mean = None
        self.std = None
        self.updatemean = True
        self.indim = indim

    def cal(self, X):
        if self.updatemean:
            tmean, tstd = np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True)
            if self.mean is None or self.std is None:
                self.mean = tmean
                self.std = tstd
            else:
                self.mean *= self.momentum
                self.mean += (1-self.momentum) * tmean
                self.std *= self.momentum
                self.std += (1-self.momentum) * tstd
        X = X.copy()
        X -= self.mean
        X /= (self.std + self.EPS)
        self.cache.append(X.copy())
        X *= self.params[0]
        X += self.params[1]
        return X

    def backcal(self, grad):
        X = self.cache[-1]
        self.grad.append(np.multiply(X, grad).reshape(-1, self.indim).sum(axis=0))
        self.grad.append(grad.reshape(-1, self.indim).sum(axis=0))
        return (grad*self.params[0])/ (self.std + self.EPS)
    
    def eval(self):
        self.updatemean = False

    def train(self):
        self.updatemean = True

class Dropout(Node):
    '''
    input shape (*)
    output (*)
    '''
    def __init__(self, p: float = 0.1):
        super().__init__("dropout")
        assert 0<=p<=1, "p 是dropout 概率，必须在[0, 1]中"
        self.p = p
        self.dropout = True

    def cal(self, X):
        if self.dropout:
            X = X.copy()
            mask = np.random.rand(*X.shape) < self.p
            np.putmask(X, mask, 0)
            X = X * (1/(1-self.p))
            self.cache.append(mask)
        return X
    
    def backcal(self, grad):
        if self.dropout:
            grad = grad.copy()
            np.putmask(grad, self.cache[-1], 0)
            grad = grad * (1/(1-self.p))
        return grad
    
    def eval(self):
        self.dropout=False

    def train(self):
        self.dropout=True

class Attention(Node):
    # input X: (B, L, d)
    # weight W_q: (d, d), W_k: (d, d), W_v: (d, d)
    # output X: (B, L, d)
    def __init__(self, dim):
        super().__init__("Attention")
        W_q = kaiming_uniform(dim, dim)
        W_k = kaiming_uniform(dim, dim)
        W_v = kaiming_uniform(dim, dim)
        self.dim = dim
        super().__init__("Attention", W_q, W_k, W_v)
        
    def cal(self, X):
        W_q, W_k, W_v = self.params[0], self.params[1], self.params[2]
        # 忽略线性映射的偏置项且需要包括残差连接
        # 即实现Attention(X, X, X)+X

        # 生成 Q, K, V
        Q = np.matmul(X, W_q) # (B, L, D)
        K = np.matmul(X, W_k)
        V = np.matmul(X, W_v)

        EPS = 1e-6
        # 获取权重
        scores = np.matmul(Q, np.transpose(K, (0, 2, 1)))
        # 缩放
        scores = scores / math.sqrt(self.dim) # (B, L, L)
        scores = scores - np.max(scores, axis = -1, keepdims = True)
        exp_scores = np.exp(scores)

        prob = exp_scores / (np.sum(exp_scores, axis = -1, keepdims = True) + EPS)
        output = np.matmul(prob, V)
        output = output + X
        
        self.cache.append((X, prob, Q, K, V))
        return output

    def backcal(self, grad):
        # 需要在前向过程中提供以下值，X:输入, softmax_score:attention中注意力权重（不是最终返回值）, Q, K, V:线性映射后Q、K、V矩阵
        X, softmax_score, Q, K, V = self.cache[-1] 
        W_q, W_k, W_v = self.params[0], self.params[1], self.params[2]
        
        # dl/dxxx
        softmax_grad = grad @ V.transpose(0, 2, 1)
        softmax_grad = np.multiply(softmax_grad, softmax_score)
        softmax_grad = softmax_grad - np.multiply(softmax_grad.sum(axis=-1, keepdims=True), softmax_score)
        
        # dl/dQ
        self.grad.append((X.transpose(0, 2, 1) @ softmax_grad @ K / np.sqrt(self.dim)).sum(0))
        # dl/dK
        self.grad.append(((Q.transpose(0, 2, 1) @ softmax_grad @ X).transpose(0, 2, 1) / np.sqrt(self.dim)).sum(0))
        # dl/dV
        self.grad.append(((softmax_score @ X).transpose(0, 2, 1) @ grad).sum(0))
        
        return grad + softmax_grad @ (W_q @ K.transpose(0, 2, 1)).transpose(0, 2, 1) / np.sqrt(self.dim) + softmax_grad.transpose(0, 2, 1) @ (Q @ W_k.T) / np.sqrt(self.dim) + softmax_score.transpose(0, 2, 1) @ grad @ W_v.T


class Linear(Node):
    # input X: (*,d1)
    # param weight: (d1, d2)
    # param bias: (*, d2) # k 层有 d1 个 neuron，k + 1 层有 d2 个neuron
    # output Linear(X): (*, d2)

    # k 层有 d1 个 neuron，k + 1 层有 d2 个neuron
    def __init__(self, indim, outdim):
        """
        初始化
        @param indim: 输入维度
        @param outdim: 输出维度
        """
        weight = kaiming_uniform(indim, outdim)
        bias = zeros(outdim)
        super().__init__("linear", weight, bias)

    def cal(self, X):
        weight = self.params[0]
        bias = self.params[1]
        ret = np.dot(X, weight) + bias
        self.cache.append(X)
        return ret

    def backcal(self, grad):
        '''
        需要保存weight和bias的梯度，可以参考Node类和BatchNorm类
        '''
        x = self.cache[-1]
        weight = self.params[0]
        # grad 与 output 的维度一致
        dw = np.dot(x.T, grad)
        db = np.sum(grad, axis = 0) # 每个 neuron 都会对 db 产生贡献
        self.grad.append(dw)
        self.grad.append(db)

        dX = np.dot(grad, self.params[0].T)
        return dX

class ResLinear(Node):
    # shape x: (*, d1)
    # shape weight: (d1, d2)
    # shape bias: (d2)
    # shape value: (*, d2) 
    def __init__(self, dim):
        """
        初始化
        @param dim: 输入, 输出维度
        """
        weight = kaiming_uniform(dim, dim)
        bias = zeros(dim)
        super().__init__("ResLinear", weight, bias)

    def cal(self, X):
        # 提示：相比于线性层，增加了残差连接
        # 即Linear(X)+X
        self.cache.append(X)
        weight, bias = self.params[0], self.params[1]
        linear_out = np.dot(X, weight) + bias
        output = linear_out + X
        return output

    def backcal(self, grad):
        X = self.cache[-1]
        weight = self.params[0]

        N, T, D = X.shape
        X_2d = X.reshape(-1, D)
        grad_2d = grad.reshape(-1, D)

        dw = np.matmul(X_2d.T, grad_2d)
        db = np.sum(grad_2d, axis = 0)
        self.grad.append(dw)
        self.grad.append(db)
        dX = np.matmul(grad_2d, weight.T) + grad_2d
        return dX

class relu(Node):
    # input X: (*)，即可能是任意维度
    # output relu(X): (*)
    def __init__(self):
        super().__init__("relu")

    def cal(self, X):
        self.cache.append(X)
        return np.clip(X, 0, None)

    def backcal(self, grad):
        return np.multiply(grad, self.cache[-1] > 0) 


class LogSoftmax(Node):
    # input X: (*)
    # output logsoftmax(X): (*), logsoftmax at 'dim'
    def __init__(self, dim=-1):
        super().__init__("logsoftmax")
        self.dim = dim

    def cal(self, X):
        EPS = 1e-8
        X = X - np.max(X, axis = self.dim, keepdims = True)
        expX = np.exp(X)
        ret = X - np.log(np.sum(expX, axis = self.dim, keepdims = True)) 
        self.cache.append(ret)
        return ret
    def backcal(self, grad):
        logsoftmaxX = self.cache[-1]
        return grad - np.multiply(np.exp(logsoftmaxX), np.sum(grad, axis = self.dim, keepdims = True))

class Sum(Node):
    def __init__(self, dim):
        super().__init__("Sum")
        self.dim = dim
    
    def cal(self, X):
        self.cache.append(X.shape[self.dim])
        return np.sum(X, axis=self.dim)
    
    def backcal(self, grad):
        return np.expand_dims(grad, axis=self.dim).repeat(self.cache[-1], axis=self.dim)
        
class Mean(Node):
    def __init__(self, dim):
        super().__init__("Mean")
        self.dim = dim
    
    def cal(self, X):
        self.cache.append(X.shape[self.dim])
        return np.mean(X, axis=self.dim)
    
    def backcal(self, grad):
        return np.expand_dims(grad, axis=self.dim).repeat(self.cache[-1], axis=self.dim) / self.cache[-1]


class NLLLoss(Node):
    '''
    negative log-likelihood 损失函数
    '''
    # shape x: (*, d), y: (*)
    # shape value: number 
    # 输入：log概率
    # 输出：NLL损失
    def __init__(self, y):
        """
        初始化
        @param y: n 样本的label
        """
        super().__init__("NLLLoss")
        self.y = y

    def cal(self, X):
        y = self.y
        self.cache.append(X)
        return - np.sum(
            np.take_along_axis(X, np.expand_dims(y, axis=-1), axis=-1))

    def backcal(self, grad):
        X, y = self.cache[-1], self.y
        ret = np.zeros_like(X)
        np.put_along_axis(ret, np.expand_dims(y, axis=-1), -1, axis=-1)
        return grad * ret