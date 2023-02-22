import random
import torch
from d2l import torch as d2l


# 根据带有噪声的线性模型构造一个人造数据集
def synthetic_data(w, b, num_example):
    """生成y = Xw + b + 噪声"""
    X = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))

    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(paramss, lr, batch_size):  # lr:learning rate 学习率
    """下批量随机梯度下降"""
    with torch.no_grad():
        for param in paramss:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print('feature:', features[0], '\nlabel:', labels[0])

    # d2l.set_figsize()
    # d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    # d2l.plt.show()

    batch_size = 10
    for X,y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break

    # 定义初始化模型参数
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    # 训练过程
    lr = 0.03
    num_epochs = 10
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X,y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)  # 'X'和'y'的小批量损失
            #  因为'l'的形状是('batch/-size', 1), 而不是一个标量。'1'中的所有元素被加到
            #  并以此计算关于['w', 'b']的梯度
            l.sum().backward()
            sgd([w, b], lr, batch_size)  # 使用参数的梯度更新
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差:{true_b - b}')