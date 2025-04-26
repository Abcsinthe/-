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

# 梯度下降参数（手动选择的超参数）
epsilon = 0.01        # 学习率（learning rate）
reg_lambda = 0.01     # 正则化强度（L2 正则项系数）

# Sigmoid 函数及其导数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)  # a 是 sigmoid 的输出

# 计算整个数据集上的总损失（用于评估模型效果）
def calculate_loss(model):
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    z1 = X.dot(W1) + b1                   # 输入层 → 隐藏层
    a1 = sigmoid(z1)                      # 激活函数：sigmoid
    z2 = a1.dot(W2) + b2                  # 隐藏层 → 输出层
    exp_scores = np.exp(z2)               # 对每个类别计算 e^score
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率分布

    # 计算交叉熵损失（对数损失）
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    data_loss += (reg_lambda / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))  # L2 正则化
    return data_loss / num_examples

# 预测函数：根据输入样本 x，输出类别（0 或 1）
def predict(model, x):
    W1, b1 = model['W1'], model['b1']
    W2, b2 = model['W2'], model['b2']
    z1 = x.dot(W1) + b1          # 输入层 → 隐藏层
    a1 = sigmoid(z1)             # 激活函数：sigmoid
    z2 = a1.dot(W2) + b2         # 隐藏层 → 输出层
    exp_scores = np.exp(z2)      # 指数函数
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # softmax 概率分布
    return np.argmax(probs, axis=1)

# 训练神经网络，学习模型参数并返回最终模型
def build_model(nn_hdim, num_passes=30000, print_loss=False):
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    model = {}

    for i in range(num_passes):
        z1 = X.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        delta3 = probs
        delta3[range(num_examples), y] -= 1         # 输出误差（预测 - 真实）
        dW2 = a1.T.dot(delta3)                      # 输出层权重梯度
        db2 = np.sum(delta3, axis=0, keepdims=True) # 输出层偏置梯度

        delta2 = delta3.dot(W2.T) * sigmoid_derivative(a1)  # 使用 sigmoid 的导数
        dW1 = X.T.dot(delta2)                             # 隐藏层权重梯度
        db1 = np.sum(delta2, axis=0)                      # 隐藏层偏置梯度

        # L2 正则化
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 参数更新（梯度下降）
        W1 -= epsilon * dW1
        b1 -= epsilon * db1
        W2 -= epsilon * dW2
        b2 -= epsilon * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print(f"迭代 {i} 次后的损失值：{calculate_loss(model):.6f}")

    return model

# 构建一个隐藏层维度为 3 的神经网络模型，并训练
model = build_model(nn_hdim=3, print_loss=True)

# 绘制决策边界
plt.figure(figsize=(8, 6))

def plot_decision_boundary(pred_func):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 填充颜色
    plt.contourf(xx, yy, Z, alpha=0.6, colors=['#a0c4ff', '#ffc9c9'], levels=1)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=60, edgecolors='k', marker='o', color='blue', label='Class 0')  # 类别 0
    plt.scatter(X[y == 1, 0], X[y == 1, 1], s=60, edgecolors='k', marker='*', color='red', label='Class 1')  # 类别 1

# 使用训练好的模型绘制决策边界
plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.legend()
plt.savefig("ai_net_img_03.png", dpi=300)
plt.show()