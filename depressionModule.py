import torch.nn as nn


class DepressionNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function='sigmoid'):
        super(DepressionNet, self).__init__()
        # 定义神经网络的层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], output_size)
        # 使用指定的激活函数
        self.activation_function = activation_function

    def forward(self, x):
        # 定义神经网络的前向传播
        x = getattr(nn.functional, self.activation_function)(self.fc1(x))
        x = getattr(nn.functional, self.activation_function)(self.fc2(x))
        x = getattr(nn.functional, self.activation_function)(self.fc3(x))
        return x