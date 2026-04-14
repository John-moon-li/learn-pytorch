import random
import torch
from d2l import torch as d2l
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul (X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
true_w=torch.tensor([2, -3.4])
true_b=4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter (features[:, 1].detach() .numpy (),
labels.detach().numpy(),1)
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
#这些样本是随机读取的,没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples) ])
        yield features[batch_indices], labels[batch_indices]
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch. zeros(1, requires_grad=True)
def linreg(X, w,b):
    return torch.matmul (X,w) + b
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape) ) ** 2 / 2
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
lr = 0.03          # 学习率（Learning Rate）：控制参数更新的步长
num_epochs = 10     # 训练轮数：把完整训练集迭代3次
net = linreg       # 绑定模型：线性回归
loss = squared_loss# 绑定损失函数：平方损失（MSE的基础形式）
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # 计算当前小批量的损失
        # 对损失求和后反向传播，计算梯度
            l.sum().backward()
        # 用SGD更新参数w和b
            sgd([w, b], lr, batch_size)
    # 每个epoch结束后，计算全训练集平均损失并打印
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# -------------------------- 8. 训练后验证参数（可选） --------------------------
print(f'\nw的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')# learn-pytorch
