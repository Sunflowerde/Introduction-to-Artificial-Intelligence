import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.04  # 学习率
wd = 2   # l2正则化项系数


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """

    X = X / 255 # 像素为 0-255，将其归一化为 0-1
    return np.dot(X, weight) + bias # 使用 np.dot() 保证数据稳定性
    raise NotImplementedError

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """

    n = X.shape[0]
    epsilon = 1e-10
    Y = (Y + 1) // 2 # 转换为{0, 1}方便计算交叉熵

    haty = predict(X, weight, bias)
    probability = sigmoid(haty)
    probability = np.clip(probability, epsilon, 1 - epsilon) # np.clip用法：将array中小于epsilon和大于1-epsilon的数字都设置为边界值，防止log计算中出现0或1

    cross_entropy = -(Y * np.log(probability) + (1 - Y) * np.log(1 - probability))
    loss = np.mean(cross_entropy) # 交叉熵损失

    l2_penalty = 0.5 * wd * np.sum(weight ** 2) # 权重平方和的惩罚项
    loss += l2_penalty

    error = probability- Y # d * 1
    grad_weight = wd * weight + (np.dot(X.T, error)) / n
    grad_bias = np.mean(error)

    weight = weight - lr * grad_weight
    bias = bias - lr * grad_bias

    return haty, loss, weight, bias
    raise NotImplementedError
