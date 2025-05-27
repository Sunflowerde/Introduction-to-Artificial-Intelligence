import numpy as np
import mnist
import pickle
import scipy.ndimage as ndimage # 图像处理
from autograd.utils import PermIterator # 生成随机全排列
from util import setseed # 固定随机数种子
from autograd.BaseGraph import Graph # 生成计算图
from autograd.BaseNode import *

setseed(0) # 便于复现
save_path = "model/myTraining.npy"

lr = 1e-3  
wd1 = 1e-5 
wd2 = 1e-5 
batchsize = 128

# 初始化数据
X = mnist.val_X
Y = mnist.val_Y

num_sample = X.shape[0]
X = X.reshape(num_sample, -1) # flatten
Y = Y.reshape(num_sample)
num_feat = X.shape[1]
num_class = np.max(Y) + 1

# 数据增强
translation = np.random.randint(-4, 5, size = (num_sample, 2)) # 水平和竖直方向的平移量
rotation = np.random.uniform(-15, 15, size = num_sample) # 旋转量
for i in range(num_sample):
  X[i] = ndimage.rotate(X[i].reshape(28, 28), rotation[i], reshape = False).flatten() # 需要先变回 28 * 28，然后再重新flatten
  shift = ndimage.shift(X[i].reshape(28, 28), translation[i])
  X[i] = shift.flatten()

# 构建计算图
def buildGraph(Y):
  nodes = [BatchNorm(784), # 归一化
           Linear(num_feat, 512), relu(),
           Dropout(0.31),
           Linear(512, 256), relu(),
           Dropout(0.26),
           Linear(256, 128), relu(),
           Dropout(0.23),
           Linear(128, 64), relu(),
           Dropout(0.18),
           Linear(64, 32), relu(),
           Dropout(0.20),
           Linear(32, num_class),
           Softmax(),
           CrossEntropyLoss(Y)]
  graph = Graph(nodes)
  return graph

if __name__ == "__main__":
  graph = buildGraph(Y)

  # 训练
  best_train_acc = 0
  dataLoader = PermIterator(num_sample, batchsize)
  graph.train()
  
  for i in range(1, 61): # epoch 数量
    hatys = [] # 存储模型预测的类别
    ys = [] # 存储真实标签
    losses = [] # 存储每个 batch 的真实损失
    graph.train()

    for perm in dataLoader: # 分批训练
      batch_X = X[perm]
      batch_Y = Y[perm]
      graph[-1].y = batch_Y

      graph.flush()
      pred, loss = graph.forward(batch_X)[-2:]
      hatys.append(np.max(pred, axis = 1))
      ys.append(batch_Y)

      graph.backward()
      graph.optimstep(lr, wd1, wd2)
      losses.append(loss)

    loss = np.mean(losses)
    acc = np.mean(np.concatenate(hatys) == np.concatenate(ys)) # np.concatenate() 将普通数组合并为长数组
    print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")

    if acc > best_train_acc:
      best_train_acc = acc

      with open(save_path, "wb") as f:
        pickle.dump(graph, f)