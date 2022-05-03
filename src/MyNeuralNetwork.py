# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 180


# Sigmoid函数
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


# 对Sigmoid函数求导
def derivative_sigmoid(x: np.ndarray) -> np.ndarray:
    fx = sigmoid(x)
    return fx * (1 - fx)


# 通过MSE求解LOSS
def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    return ((y_true - y_pred) ** 2).mean()


# 神经网络类
class MyNeuralNetwork:
    def __init__(self):
        # 权重
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # 偏差
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()
        # Loss
        self.loss_history = []

    def feedforward(self, x):
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data: np.ndarray, all_y_trues: np.ndarray):
        learn_rate = 0.1
        epochs = 1000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # 向前传播
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1
                # 计算偏导数
                d_l_d_ypred = -2 * (y_true - y_pred)
                # 神经元o1
                d_ypred_d_w5 = h1 * derivative_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * derivative_sigmoid(sum_o1)
                d_ypred_d_b3 = derivative_sigmoid(sum_o1)
                d_ypred_d_h1 = self.w5 * derivative_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * derivative_sigmoid(sum_o1)
                # 神经元h1
                d_h1_d_w1 = x[0] * derivative_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * derivative_sigmoid(sum_h1)
                d_h1_d_b1 = derivative_sigmoid(sum_h2)
                # 神经元h2
                d_h2_d_w3 = x[0] * derivative_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * derivative_sigmoid(sum_h2)
                d_h2_d_b2 = derivative_sigmoid(sum_h2)
                # 更新权重和偏差
                # 神经元h1
                self.w1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_l_d_ypred * d_ypred_d_h1 * d_h1_d_b1
                # 神经元h2
                self.w3 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_l_d_ypred * d_ypred_d_h2 * d_h2_d_b2
                # 神经元o1
                self.w5 -= learn_rate * d_l_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_l_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_l_d_ypred * d_ypred_d_b3
            if epoch % 10 == 0:
                # 相当于 y_preds = [self.feedforward(i) for i in data]
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                self.loss_history.append(loss)
                print("Epoch {:d} \tloss: {:.3f}".format(epoch, loss))


# 定义数据集
data = np.array([
    [-2, -1],
    [25, 6],
    [17, 4],
    [-15, -6],
])

all_y_trues = np.array([
    1,
    0,
    0,
    1,
])

# 训练神经网络
network = MyNeuralNetwork()
network.train(data, all_y_trues)

# 绘制Loss曲线
plt.figure()
plt.plot(np.linspace(0, 1000, 100), network.loss_history)
plt.title("神经网络Loss变化图")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("../images/my-neural-network-loss.png")
plt.show()
