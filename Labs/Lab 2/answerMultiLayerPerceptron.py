import mnist
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *

# 超参数
# TODO: You can change the hyperparameters here
lr = 5e-4   # 学习率
wd1 = 1e-5  # L1正则化
wd2 = 1e-6  # L2正则化
batchsize = 128

def buildGraph(Y):
    """
    建图
    @param Y: n 样本的label
    @return: Graph类的实例, 建好的图
    """
    nodes = [StdScaler(mnist.mean_X, mnist.std_X), # 数据初始化
             Linear(mnist.num_feat, 1024), relu(), # Linear 层，线性层后接非线性层
             Linear(1024, 1024), relu(), Dropout(0.25), # dropout 防止过拟合
             Linear(1024, 1024), relu(), Dropout(0.3),
             Linear(1024, 1024), relu(), Dropout(0.21),
             Linear(1024, mnist.num_class), Softmax(), # 输出维度为 num_class，用 softmax 计算概率
             CrossEntropyLoss(Y)] # 最后计算损失函数  
    graph = Graph(nodes)
    return graph
