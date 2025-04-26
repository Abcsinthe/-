import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Agg'
from sklearn.datasets import make_moons

# 生成双月形数据集
np.random.seed(0)  # 设置随机种子以保证结果可复现
X, y = make_moons(n_samples=200, noise=0.2)

# 训练集大小
num_examples = len(X)

# 输入层维度（二维坐标输入）
nn_input_dim = 2

# 输出层维度（2个类别，使用 one-hot 编码）
nn_output_dim = 2

# 梯度下降参数
epsilon = 0.01        # 学习率（learning rate）
reg_lambda = 0.01     # 正则化强度（L2 正则项系数）

# Sigmoid 函数及其导数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# 训练神经网络，学习模型参数并返回最终模型
def build_model(nn_hdim, num_passes=30000):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    for i in range(num_passes):
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = a1.T.dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        delta2 = delta3.dot(W2.T) * sigmoid_derivative(a1)
        dW1 = X.T.dot(delta2)
        db1 = np.sum(delta2, axis=0)

        # L2 正则化
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 参数更新
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

    model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

# 绘制决策边界
def plot_decision_boundary(pred_func, ax, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.6, colors=['#a0c4ff', '#ffc9c9'], levels=1)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=60, edgecolors='k', marker='o', color='blue', label='Class 0')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=60, edgecolors='k', marker='*', color='red', label='Class 1')
    ax.set_title(title)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.grid(True)

# 创建图形以显示不同神经元数量的决策边界
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Decision Boundaries with Different Neuron Counts")

# 训练并绘制每个模型
for i in range(1, 8):
    model = build_model(nn_hdim=i)
    plot_decision_boundary(lambda x: np.argmax(sigmoid(x.dot(model['W1']) + model['b1']).dot(model['W2']) + model['b2'], axis=1),
                           axs[(i-1)//3, (i-1)%3], f"{i} Neurons")

# 隐藏最后一个子图
axs[2, 2].axis('off')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.legend(loc='upper right')
plt.show()