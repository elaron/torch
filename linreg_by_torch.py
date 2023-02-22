import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


if __name__ == '__main__':
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    next(iter(data_iter))

    # Sequential 可以理解为list of layers，其实直接用net = nn.Linear(2, 1)也可以
    net = nn.Sequential(nn.Linear(2, 1))

    # 初始化模型参数
    # 0表示访问哪一层，weight表示w，data表示真实data，normal_表示使用正态分布替换data的值
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    # net.parameters()包括了lingre模型的所有参数，包括w和b
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
