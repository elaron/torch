import torch


# 试一下axis求和
def cal4():
    x = torch.ones(48).reshape(2,3,4,2)
    print(x)

    y = x.sum(axis=1, keepdims=True)
    z = x.sum(axis=1)
    print(y, y.shape)
    print(z, z.shape)


# 控制流的梯度计算
def cal3(a):
    a = torch.randn(size=(), requires_grad=True)
    d = f(a)
    d.backward()


def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


# 分离计算
def cal2():
    x = torch.arange(4.0, requires_grad=True)
    y = x * x
    u = y.detach()
    z = u * x

    z.sum().backward()
    print(x.grad == u)


def cal():
    # 创建变量x
    x = torch.arange(4.0, requires_grad=True)
    print(x)
    print(x.grad)

    # 定义含y
    y = 2 * torch.dot(x, x)
    print(y)

    # 计算y关于每个x分量的梯度
    y.backward()
    print(x.grad)

    # 计算另一个函数
    x.grad.zero_()  # 默认情况下，pytorch会累积梯度，需要先清零一下
    y = x.sum()
    y.backward()
    print(x.grad)


if __name__ == '__main__':
    # cal()
    cal4()