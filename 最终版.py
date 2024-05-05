import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


# 定义神经网络类
class CustomNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation_function='sigmoid'):
        super(CustomNeuralNetwork, self).__init__()
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

# 定义自定义损失函数
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        # 计算平方误差损失
        return torch.mean((output - target) ** 2)
# 超参数
input_size = 22  # 输入的特征维度为 22
hidden_sizes = [128, 100]  # 隐藏层的大小
output_size = 1  # 输出的类别数量
learning_rate = 0.01
epochs = 1024
batch_size = 32  # 定义批次大小

# 构建模型
model = CustomNeuralNetwork(input_size, hidden_sizes, output_size, activation_function='relu')
criterion = CustomLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 读取Excel文件
train_data = pd.read_csv('dataset/train.csv')

# 提取特征和标签
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# 将数据转换为PyTorch张量
X_tensor = torch.tensor(X_train, dtype=torch.float32)
y_tensor = torch.tensor(y_train, dtype=torch.float32)

# 创建TensorDataset和DataLoader
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(epochs):
    model.train()
    loss, current, n = 0.0, 0.0, 0  # 初始化损失、准确率和批次计数器
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()  # 清空梯度
        outputs = model(X_batch) # 前向传播
        current_loss = criterion(outputs, y_batch)

        current_loss.backward()
        optimizer.step()

        loss += current_loss.item()  # 累积损失
        # current += cur_epoch_acc.item()  # 累积准确率
        n += 1  # 增加批次计数器
    # 打印进度

    if (epoch + 1) % 100 == 0 :
        print(f"Epoch [{epoch + 1}/{epochs}], Batch {batch_idx}, Loss: {loss:.4f}")

# 读取测试数据
test_data = pd.read_csv('dataset/test.csv')

# 提取特征和标签
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

# 将测试数据转换为PyTorch张量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# 使用训练好的模型进行预测
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()

# 输出预测值和真实值
for i, (pred, true) in enumerate(zip(predictions, y_test)):
    if pred > 0.5:
        print(f"[第{i+1}例，预测值：{1}, 真实值：{int(true)}, {'true' if int(pred) == int(true) else 'false'}]")
    else:
        print(f"[第{i+1}例，预测值：{0}, 真实值：{int(true)}, {'true' if int(pred) == int(true) else 'false'}]")
# 计算准确率
accuracy = np.mean((predictions > 0.5) == y_test)
print(f"Accuracy: {accuracy}")
