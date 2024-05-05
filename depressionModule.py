import torch.nn as nn
import torch


class DepressionNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DepressionNet, self).__init__()
        # 定义一个将输入特征数映射到第一个隐藏层的全连接层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        # 定义隐藏层之间的全连接层
        layers = []
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())  # 在每个隐藏层后添加ReLU激活函数
        self.layers = nn.Sequential(*layers)  # 使用Sequential容器将多个层组合起来

        # 定义最后一个全连接层，将最后一个隐藏层映射到输出
        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)
        self.sigmoid = nn.Sigmoid()  # 添加Sigmoid激活函数
        # 输出层使用sigmoid激活函数
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # 定义前向传播过程
        x = self.fc1(x)
        x = self.layers(x)  # 通过所有隐藏层
        x = self.fc2(x)  # 通过最后的全连接层
        x = self.output_activation(x)  # 应用输出层激活函数
        return self.sigmoid(x)