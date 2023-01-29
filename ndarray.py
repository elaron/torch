import torch


def create_array():
    x = torch.arange(12)
    print(x)  # 生成一个一维的张量
    print(x.shape)  # 查看张量的维度
    print(x.numel())  # 查看张量的元素个数
    print(x.reshape(3, 4))  # 改变张量的形状为3*4矩阵
    print(x.reshape(-1, 2))  # 改变张量的形状为2列的矩阵
    print(x.reshape(2, -1))  # 改变张量的形状为2行的矩阵
    print(torch.zeros(3, 4))  # 创建一个3*4的0矩阵
    print(torch.ones(3, 4))  # 创建一个3*4的1矩阵
    print(torch.randn(3, 4))  # 创建一个3*4的随机矩阵
    print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))  # 指定初始值来创建矩阵


def cal_each_elem():
    x = torch.tensor([1.0, 2, 4, 8])
    y = torch.tensor([2, 2, 2, 2])
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)  # **运算符是求幂运算
    print(torch.exp(x))


def tensor_cat():
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(torch.cat((x, y), dim=0))  # 在0维度上拼接x和y张量
    print(torch.cat((x, y), dim=1))  # 在1维度上拼接x和y张量


def bool_cal():
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print(x == y)


def sum():
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    print(x.sum())


def broadcasting_mechanism():
    x = torch.arange(3).reshape((3,1))
    y = torch.arange(2).reshape((1, 2))
    print(x, y)
    print(x + y)  # 通过广播机制将两个矩阵扩大为3*2的矩阵


def index_tensor():
    x = torch.arange(12).reshape((3,4))
    print(x)
    print(x[0])  # 取tensor的第一个元素
    print(x[-1])  # 取tensor的最一个元素
    print(x[1:3])  # 取tensor的索引为[n:m)之间的元素

    x[1, 2] = 100  # 更新指定位置元素的值
    print(x)

    x[0:2, :] = 12  # 第0：2行及所有列的值都赋为12
    print(x)


def save_memory():
    # x = x + y 时，x会创建一片新的内存用于存放计算后的结果
    x = torch.arange(12)
    y = torch.ones(12)
    before = id(y)
    y = x + y
    after = id(y)
    print(before == after)

    # 若想节省内存，可以使用 x[:] = x + y或y+=x，这样torch就会在原地进行计算和存储，不会开辟新的内存
    before = id(y)
    y[:] = x + y
    after = id(y)
    print(before == after)

    before = id(y)
    y += x
    after = id(y)
    print(before == after)


def type_trans():
    x = torch.arange(12)
    a = x.numpy()
    b = torch.tensor(a)
    print(type(a), type(b))  # tensor和numpy张量类型互转

    a = torch.tensor([3.5])
    print(a, a.item(), float(a), int(a))  # 将tensor转换为numpy类型


if __name__ == '__main__':
    # create_array()  # 多种创建张量的方法
    # cal_each_elem()  # 按元素计算
    # tensor_cat()  # 张量的连结
    # bool_cal()  # 通过逻辑运算符构建二元张量
    # sum()  # 对张量的元素求和
    # broadcasting_mechanism()  # 广播机制
    # index_tensor()  # 按索引读取或更新张量的值
    # save_memory()  # 节省内存的方法
    type_trans()
