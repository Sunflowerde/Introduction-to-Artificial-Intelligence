import numpy as np
from copy import deepcopy
from typing import List, Callable

EPS = 1e-6

# 超参数，分别为树的最大深度、熵的阈值、信息增益函数
# TODO: You can change or add the hyperparameters here
hyperparams = {
    "depth": 4, 
    "purity_bound": 0.1, 
    "gainfunc": "gainratio" 
    }

def entropy(Y: np.ndarray):
    """
    计算熵
    @param Y: (n,), 标签向量
    @return: 熵
    """

    _, counts = np.unique(Y, return_counts = True) # np.unique()用法：返回 Y 中出现的标签及每个标签出现的次数
    n = len(Y)

    prob = counts / n # 计算每个标签出现的概率
    prob = np.clip(prob, EPS, 1 - EPS) # 防止 log2 溢出
    entropy_value = -np.sum(prob * np.log2(prob))
    return entropy_value
    
    raise NotImplementedError


def gain(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益
    """
    feat = X[:, idx]
    values, counts = np.unique(feat, return_counts = True)
    n = len(Y)
    conditional_entropy = 0

    for value, count in zip(values, counts): # zip() 用法：将两个列表中相同位置的元素一一对应
        mask = (feat == value) # 找出 feat 中与 value 相同的元素的位置
        subset = Y[mask] # 找到满足 feat 的子集
        conditional_entropy += count / n * entropy(subset)

    return entropy(Y) - conditional_entropy
    return NotImplementedError


def gainratio(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算信息增益比
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 信息增益比
    """
    ret = gain(X, Y, idx) / (entropy(X[:, idx]) + EPS)
    return ret


def giniD(Y: np.ndarray):
    """
    计算基尼指数
    @param Y: (n,), 样本的label
    @return: 基尼指数
    """
    u, cnt = np.unique(Y, return_counts=True)
    p = cnt / Y.shape[0]
    return 1 - np.sum(np.multiply(p, p))


def negginiDA(X: np.ndarray, Y: np.ndarray, idx: int):
    """
    计算负的基尼指数增益
    @param X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: (n,), 样本的label
    @param idx: 第idx个特征
    @return: 负的基尼指数增益
    """
    feat = X[:, idx]
    ufeat, featcnt = np.unique(feat, return_counts=True)
    featp = featcnt / feat.shape[0]
    ret = 0
    for i, u in enumerate(ufeat):
        mask = (feat == u)
        ret -= featp[i] * giniD(Y[mask])
    ret += giniD(Y)  # 调整为正值，便于比较
    return ret


class Node:
    """
    决策树中使用的节点类
    """
    def __init__(self): 
        self.children = {}          # 子节点列表，其中key是特征的取值，value是子节点（Node）
        self.featidx: int = None    # 用于划分的特征
        self.label: int = None      # 叶节点的标签

    def isLeaf(self):
        """
        判断是否为叶节点
        @return: bool, 是否为叶节点
        """
        return len(self.children) == 0


def buildTree(X: np.ndarray, Y: np.ndarray, unused: List[int], depth: int, purity_bound: float, gainfunc: Callable, prefixstr=""):
    """
    递归构建决策树。
    @params X: (n, d), 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @params Y: (n,), 样本的label
    @params unused: List of int, 未使用的特征索引
    @params depth: int, 树的当前深度
    @params purity_bound: float, 熵的阈值
    @params gainfunc: Callable, 信息增益函数
    @params prefixstr: str, 用于打印决策树结构
    @return: Node, 决策树的根节点
    """
    
    root = Node()
    u, ucnt = np.unique(Y, return_counts=True) # 返回样本的标签及每个标签的个数
    root.label = u[np.argmax(ucnt)]
    # print(prefixstr, f"label {root.label} numbers {u} count {ucnt}") #可用于debug
    # 当达到终止条件时，返回叶节点
    # 3 个终止条件：1.所有样本标签相同 2.没有可用特征 3.属性增益过小
    gains = [gainfunc(X, Y, i) for i in unused]
    if len(u) == 1 or not unused or np.max(gains) <= purity_bound:
        return root
    
    idx = np.argmax(gains) # 选择增益最大的标签
    root.featidx = unused[idx]
    unused = deepcopy(unused)
    unused.pop(idx)

    feat = X[:, root.featidx]
    ufeat = np.unique(feat) # 获取当前特征的取值
    root.children = {}

    for value in ufeat:
        mask = (feat == value)
        X_sub, Y_sub = X[mask], Y[mask]

        child_prefix = prefixstr + '| '
        child = buildTree(X_sub, Y_sub, unused, depth + 1, purity_bound, gainfunc, child_prefix) # 递归
        root.children[value] = child
    # 按选择的属性划分样本集，递归构建决策树
    # 提示：可以使用prefixstr来打印决策树的结构
    
    return root


def inferTree(root: Node, x: np.ndarray):
    """
    利用建好的决策树预测输入样本为哪个数字
    @param root: 当前推理节点
    @param x: d*1 单个输入样本
    @return: int 输入样本的预测值
    """
    if root.isLeaf():
        return root.label
    child = root.children.get(x[root.featidx], None)
    return root.label if child is None else inferTree(child, x)

