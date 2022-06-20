import numpy as np
import pandas as pd


def sigmoid(v):
    output = 1 / (1 + np.exp(-v))   # 激活函数
    # print(output)
    return output


def softmax(x):
    ex = np.exp(x)
    y = ex/np.sum(ex)
    return y


def create_dataset(datapath1, datapath2, datapath3):
    """数据载入"""
    x1 = pd.read_excel(datapath1, usecols=[1, 2, 3, 4])
    x2 = pd.read_excel(datapath2, usecols=[1, 2, 3, 4])
    x3 = pd.read_excel(datapath3, usecols=[1, 2, 3, 4])
    # print(type(x))
    state = np.random.get_state()
    x1 = x1.values/10000
    x2 = x2.values/10000
    x3 = x3.values / 10000
    x = np.concatenate((x1, x2, x3))
    np.random.shuffle(x)    # 顺序打乱
    # print(x[0])
    y1 = pd.read_excel(datapath1, usecols=[5])
    y2 = pd.read_excel(datapath2, usecols=[5])
    y3 = pd.read_excel(datapath3, usecols=[5])
    y1 = y1.values
    y2 = y2.values
    y3 = y3.values
    y = np.concatenate((y1, y2, y3))
    # print(y.shape)
    np.random.set_state(state)
    np.random.shuffle(y)
    x_train = x[:int(x.shape[0]*0.7)]
    x_test = x[int(x.shape[0]*0.7):]
    y_train = y[:int(x.shape[0]*0.7)]
    y_test = y[int(x.shape[0]*0.7):]
    # print(x_train.shape)
    # print(y_train.shape)
    return x_train, x_test, y_train, y_test


def NN(W1, W2, data_x, data_y, epoch, lr):
    for n_epoch in range(epoch):
        for i in range(x_train.shape[0]):
            x = data_x[i].reshape(4, 1)
            y = data_y[i]
            v1 = np.dot(x.transpose(), W1)
            # print(v1)
            y1 = sigmoid(v1)
            # print(y1)
            v2 = np.dot(y1, W2)
            # print(v2)
            y_hat = sigmoid(v2)
            # print(y_hat)
            # y_hat = softmax(v2)
            # print(y_hat)
            """back propagation"""
            e = y - y_hat
            # print(e.shape)
            e1 = np.dot(e, W2.transpose())
            delta1 = y1*(1-y1)*e1
            # print(delta1.shape)
            dW1 = np.dot(data_x[i].reshape(len(data_x[i].flatten()), 1), lr * delta1)
            W1 = W1 + dW1
            W2 = W2 + lr * y_hat * (1 - y_hat) * e * y1.transpose()
    return W1, W2


print('课程任务3--回归任务')
datapath1 = 'CS2_33.xlsx'
datapath2 = 'CS2_34.xlsx'
datapath3 = 'CS2_35.xlsx'

x_train, x_test, y_train, y_test = create_dataset(datapath1, datapath2, datapath3)

W1 = np.random.randn(4, 16)
W2 = np.random.randn(16, 1)

W1, W2 = NN(W1, W2, x_train, y_train, epoch=1000, lr=0.9)
loss = 0

"""测试集"""
for i in range(x_test.shape[0]):
    x = x_test[i].reshape(4, 1)
    v1 = np.dot(x.transpose(), W1)
    y1 = sigmoid(v1)
    v = np.dot(y1, W2)
    y_hat = sigmoid(v)
    loss = y_test[i] - y_hat + loss
    print('real:', y_test[i], "predict:", y_hat)
print('Loss:', loss/x_test.shape[0])    # 计算预测值和真实值之间的平均误差
