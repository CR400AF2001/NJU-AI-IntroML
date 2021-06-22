import numpy as np
seed = 1
np.random.seed(seed)
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)，x可以是标量、向量或矩阵）
    return 1 / (1 + np.exp(-x))
    

def deriv_sigmoid(x):
    # （需要填写的地方，输入x返回sigmoid(x)在x点的梯度，x可以是标量、向量或矩阵）
    return sigmoid(x) * (1 - sigmoid(x))
    

def mse_loss(y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回他们的MSE（均方误差，不需要除以2）,其中真实标记和预测值维度都是(n_samples,) 或 (n_samples, n_outputs)）
    return np.mean(np.sum(np.square(y_true - y_pred), axis=1))


    

def accuracy(y_true, y_pred):
    # （需要填写的地方，输入真实标记和预测值返回Accuracy，其中真实标记和预测值是维度相同的向量）
    y_one = y_true - y_pred
    y_one[y_one != 0] = 1
    return 1 - (np.sum(y_one) / y_one.shape[0])
    

def to_onehot(y):
    # 输入为向量，转为onehot编码
    y = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    return y

class NeuralNetwork():
    def __init__(self, d, q, l):
        # weights
        self.v = np.random.randn(d, q)
        self.w = np.random.randn(q, l)
        # biases
        self.gamma = np.random.randn(q)
        self.theta = np.random.randn(l)
        # 以上为神经网络中的权重和偏置，其中具体含义见西瓜书P101

    def predict(self, X):
        '''
        X: shape (n_samples, d)
        returns: shape (n_samples, l)
        '''
        # （需要填写的地方，输入样本，输出神经网络最后一层的输出值）
        return sigmoid(np.dot(sigmoid(np.dot(X, self.v) - np.array([self.gamma])), self.w) - np.array([self.theta]))

    
    def train(self, X, y, learning_rate = 1, epochs = 500):
        '''
        X: shape (n_samples, d)
        y: shape (n_samples, l)
        输入样本和训练标记，进行网络训练
        '''
        for epoch in range(epochs):
            # （以下部分为向前传播过程，请完成）
            h = np.dot(X, self.v) - np.array([self.gamma])
            o = np.dot(sigmoid(h), self.w) - np.array([self.theta])
            y_pre = sigmoid(o)
            # （以下部分为计算梯度，请完成）
            de_dy = y_pre - y
            # 输出层梯度
            g = deriv_sigmoid(o) * de_dy
            de_dtheta = -1 * np.mean(g, axis=0)
            de_dw = np.dot(sigmoid(h).T, g) / X.shape[0]
            # 隐层梯度
            e = deriv_sigmoid(h) * np.dot(g, self.w.T)
            de_dgamma = -1 * np.mean(e, axis=0)
            de_dv = np.dot(X.T, e) / X.shape[0]
            # 更新权重和偏置
            self.w -= learning_rate * de_dw
            self.theta -= learning_rate * de_dtheta
            self.v -= learning_rate * de_dv
            self.gamma -= learning_rate * de_dgamma
            # 计算epoch的loss
            if epoch % 10 == 0:
                y_preds = self.predict(X)
                loss = mse_loss(y, y_preds)
                print("Epoch %d loss: %.3f"%(epoch, loss))
    
if __name__ ==  '__main__':
    # 获取数据集，训练集处理成one-hot编码
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    y_train = to_onehot(y_train)

    # 训练网络（可以对下面的n_hidden_layer_size，learning_rate和epochs进行修改，以取得更好性能）
    n_features = X.shape[1]
    n_hidden_layer_size = 55
    n_outputs = len(np.unique(y))
    network = NeuralNetwork(d = n_features, q = n_hidden_layer_size, l = n_outputs)
    network.train(X_train, y_train, learning_rate = 1, epochs = 5000)

    # 预测结果
    y_pred = network.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    mse = mse_loss(to_onehot(y_test), y_pred)
    print("\nTesting MSE: {:.3f}".format(mse))
    acc = accuracy(y_test, y_pred_class) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))
    