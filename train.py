import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import depressionModule as module
import getData as DATA

# 超参数
if __name__ == '__main__':
    num = 10
    for i in range(num):

        #
        # 超参数
        input_size = 22  # 输入的特征维度为 22，与测试数据的特征维度一致
        # hidden_sizes = [128, 64, 32, 16, 8, 1]  # 隐藏层的大小
        hidden_sizes = [128, 100]  # 隐藏层的大小
        output_size = 1  # 输出的类别数量

        learning_rate = 0.01  # 优化器的学习率
        epochs = 1000  # 训练总轮数
        batch_size = 32  # 每批次的样本数量

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动判断是否使用cpu

        dataset = DATA.DataSet()
        train_dataset = dataset.get_trainset()

        # 加载训练数据
        data = pd.read_csv(train_dataset)
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values

        # 将数据转换成张量
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # 实例化模型
        model = module.DepressionNet(input_size, hidden_sizes, output_size).to(device)

        # 定义损失函数
        # 二元分类损失函数
        criterion = nn.BCELoss().to(device)

        # 定义优化器
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        def train(dataloader, model, criterion, optimizer):
            model.train()
            loss, current, n = 0.0, 0.0, 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()

                outputs = model(inputs)  # 前向传播
                if torch.isnan(inputs).any():
                    continue
                targets = targets.view(-1, 1).float()
                cur_loss = criterion(outputs, F.sigmoid(targets))  # 当前批次的损失

                _, predicted = torch.max(outputs, dim=1)  # 得到预测结果
                # print(outputs.shape[0])
                sum = 0
                for i in range(predicted.shape[0]):
                    if int(predicted[i]) == int(targets[i]):
                        sum += 1
                cur_epoch_acc = sum / outputs.shape[0]  # 计算当前批次准确率

                cur_loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                loss += cur_loss.item()  # 累计损失
                current += cur_epoch_acc  # 累计准确率
                n += 1
            return current / n, loss / n


        max_acc = 0
        for epoch in range(epochs):

            a, l = train(dataloader, model, criterion, optimizer)
            if a > max_acc:
                folder = 'models'
                if not os.path.exists(folder):
                    os.makedirs(folder)
                max_acc = a
                print(f"Saving model to {folder}")
                torch.save(model.state_dict(), f'{folder}/model_acc_{a}_loss{l}.pth')
            if (epoch + 1) % 10 == 0:
                print(f"#----------Epoch {epoch + 1}----------#")
                print(f"train loss: {l}")  # 打印平均损失
                print(f"train acc : {a}")  # 打印平均准确率
                print(f"  max acc : {max_acc}")


