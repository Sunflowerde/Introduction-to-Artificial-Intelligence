import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 50     # 树的数量
ratio_data = 1   # 采样的数据比例
ratio_feat = 0.8 # 采样的特征比例
hyperparams = {
    "depth":5, 
    "purity_bound":1e-2,
    "gainfunc": gainratio
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """

    trees = []
    n, _ = X.shape

    for _ in range(num_tree): # 样本扰动，从训练集中随机取样子集
        sample_indices = np.random.choice(n, size = int(n * ratio_data), replace = False) # 随机抽取 n * ratio_data 个索引
        X_sub, Y_sub = X[sample_indices], Y[sample_indices]

        # 属性扰动，随机选取特征
        feat_indices = np.random.choice(mnist.num_feat, size = int(mnist.num_data * ratio_feat), replace = False) # 注意这里参数选择 replace = False，为了保证样本没有重复，增加树的多样性
        feat_indices = list(feat_indices)

        # 开始构建子树
        depth = hyperparams["depth"]
        purity_bound = hyperparams["purity_bound"]
        gainfunc = hyperparams["gainfunc"]

        tree = buildTree(X_sub, Y_sub, feat_indices, depth, purity_bound, gainfunc)
        trees.append(tree)
    return trees

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]