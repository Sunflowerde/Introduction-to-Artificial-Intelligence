import math
from SST_2.dataset import traindataset, minitraindataset
from fruit import get_document, tokenize
import pickle
import numpy as np
from importlib.machinery import SourcelessFileLoader
from autograd.BaseGraph import Graph
from autograd.BaseNode import *

class NullModel:
    def __init__(self):
        pass

    def __call__(self, text):
        return 0


class NaiveBayesModel:
    def __init__(self):
        self.dataset = traindataset() # 完整训练集，需较长加载时间
        #self.dataset = minitraindataset() # 用来调试的小训练集，仅用于检查代码语法正确性

        # 以下内容可根据需要自行修改，不修改也可以完成本题
        self.token_num = [{}, {}] # token在正负样本中出现次数
        self.V = 0 #语料库token数量
        self.pos_neg_num = [0, 0] # 正负样本数量
        self.count()

    def count(self):
        # 提示：统计token分布不需要返回值
        vocabulary = set()
        for text, label in self.dataset:
            label = int(label)
            self.pos_neg_num[label] += 1 # 1 代表正类，0 代表负类

            tokens = tokenize(text)
            for token in tokens:
                vocabulary.add(token)

                if token in self.token_num[label]:
                    self.token_num[label][token] += 1 # self.token_num[label] 是一个字典，统计每个 token 的数目
                else:
                    self.token_num[label][token] = 1

        self.V = len(vocabulary)

    def __call__(self, text):
        # 返回1或0代表当前句子分类为正/负样本
        # P(y|x) = P(y) * \prod_{i=1}^n P(x_i|y)
        # P(y) 为先验概率，表示在已知句子中所有 token 的条件下，判断句子类别，正类即为正类 token 数除以总样本数
        # 然后计算每个 token 在 y 下的概率
        tokens = tokenize(text)
        log_prob = [0, 0]

        for label in [0, 1]:
            total_cnt = sum(self.token_num[label].values())
            log_prob[label] = math.log(total_cnt / sum(self.pos_neg_num))

            for token in tokens:
                cnt = self.token_num[label].get(token, 0) # 更安全的写法，default 为 0，防止出现未存储的 token
                log_prob[label] += math.log((cnt + 1) / (self.V + total_cnt))

        return 1 if log_prob[1] > log_prob[0] else 0

def buildGraph(dim, num_classes, L): #dim: 输入一维向量长度, num_classes:分类数
    # 以下类均需要在BaseNode.py中实现
    # 也可自行修改模型结构
    nodes = [Attention(dim), relu(), Dropout(0.28), LayerNorm((L, dim)), ResLinear(dim), relu(), Dropout(0.19), LayerNorm((L, dim)), Mean(1), Linear(dim, num_classes), LogSoftmax(), NLLLoss(num_classes)]
    
    graph = Graph(nodes)
    return graph


save_path = "model/attention.npy"

class Embedding():
    def __init__(self):
        self.emb = dict() 
        with open("words.txt", encoding='utf-8') as f: #word.txt存储了每个token对应的feature向量，self.emb是一个存储了token-feature键值对的Dict()，可直接调用使用
            for i in range(50000):
                row = next(f).split()
                word = row[0]
                vector = np.array([float(x) for x in row[1:]])
                self.emb[word] = vector
                if i == 0:
                    self.emb_dim = len(vector)
        
    def __call__(self, text, max_len=50):
        # 利用self.emb将句子映射为一个二维向量（LxD），注意，同时需要修改训练代码中的网络维度部分
        # 默认长度L为50，特征维度D为100
        # 提示: 考虑句子如何对齐长度，且可能存在空句子情况（即所有单词均不在emd表内） 
        # 先提取句子的所有 token
        # 然后将每个 token 的向量放入矩阵中，不足的部分补零
        tokens = tokenize(text)
        start_emb = np.zeros((max_len, self.emb_dim), dtype = np.float32)

        for i, token in enumerate(tokens):
            if i >= max_len:
                break
            if token in self.emb:
                start_emb[i] = self.emb[token]
            else:
                start_emb[i] = np.zeros((1, self.emb_dim), dtype = np.float32)
                
        return start_emb

class AttentionModel():
    def __init__(self):
        self.embedding = Embedding()
        with open(save_path, "rb") as f:
            self.network = pickle.load(f)
        self.network.eval()
        self.network.flush()

    def __call__(self, text, max_len=50):
        X = self.embedding(text, max_len)
        X = np.expand_dims(X, 0)
        pred = self.network.forward(X, removelossnode=1)[-1]
        haty = np.argmax(pred, axis=-1)
        return haty[0]


class QAModel():
    def __init__(self):
        self.document_list = get_document()

    def tf(self, word, document):
        # 返回单词在文档中的频度
        # document变量结构请参考fruit.py中get_document()函数
        p = 0
        word_cnt = document.count(word)
        total_cnt = len(document)
        p = word_cnt / total_cnt
        return p

    def idf(self, word):
        # 返回单词IDF值，提示：你需要利用self.document_list来遍历所有文档
        # 注意python整除与整数除法的区别
        document_list = self.document_list
        total_cnt = len(document_list)
        document_cnt = 0
        for document in document_list:
            if word in document["document"]:
                document_cnt += 1
        ret = math.log(total_cnt / (document_cnt + 1))
        return ret
    
    def tfidf(self, word, document):
        # 返回TF-IDF值
        return self.tf(word, document) * self.idf(word)

    def __call__(self, query):
        query = tokenize(query) # 将问题token化
        # 利用上述函数来实现QA
        # 提示：你需要根据TF-IDF值来选择一个最合适的文档，再根据IDF值选择最合适的句子
        # 返回时请返回原本句子，而不是token化后的句子，可以参考README中数据结构部分以及fruit.py中用于数据处理的get_document()函数
        best_doc = None
        max_score1 = 0
        for doc in self.document_list:
            tokens = doc["document"]
            scores = sum(self.tfidf(word, tokens) for word in query)
            if scores > max_score1:
                max_score1 = scores
                best_doc = doc
        
        best_sentence = None
        max_score2 = 0
        for tokens, sentence in best_doc["sentences"]:
            scores = sum(self.idf(word) for word in query if word in tokens)
            if scores > max_score2:
                max_score2 = scores
                best_sentence = sentence
        
        return best_sentence

modeldict = {
    "Null": NullModel,
    "Naive": NaiveBayesModel,
    "Attn": AttentionModel,
    "QA": QAModel,
}


if __name__ == '__main__':
    embedding = Embedding()
    lr = 1e-3   # 学习率
    wd1 = 1e-4  # L1正则化
    wd2 = 1e-3  # L2正则化
    batchsize = 64
    max_epoch = 10
    
    max_L = 50
    num_classes = 2
    feature_D = 100
    
    graph = buildGraph(feature_D, num_classes, max_L) # 维度可以自行修改

    # 训练
    # 完整训练集训练有点慢
    best_train_acc = 0
    dataloader = traindataset(shuffle=True) # 完整训练集
    #dataloader = minitraindataset(shuffle=True) # 用来调试的小训练集
    for i in range(1, max_epoch+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        X = []
        Y = []
        cnt = 0
        for text, label in dataloader:
            x = embedding(text, max_L)
            label = np.zeros((1)).astype(np.int32) + label
            X.append(x)
            Y.append(label)
            cnt += 1
            if cnt == batchsize:
                X = np.stack(X, 0)
                Y = np.concatenate(Y, 0)
                graph[-1].y = Y
                graph.flush()
                pred, loss = graph.forward(X)[-2:]
                hatys.append(np.argmax(pred, axis=-1))
                ys.append(Y)
                graph.backward()
                graph.optimstep(lr, wd1, wd2)
                losss.append(loss)
                cnt = 0
                X = []
                Y = []

        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)