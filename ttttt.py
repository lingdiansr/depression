import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from depressionModule import DepressionNet
import pandas as pd

# 读取CSV文件
dataframe = pd.read_csv('dataset/b_depressed.csv')

# 数据预处理
# 假设我们只选择部分特征进行训练，并且标签是最后一列'depressed'
features = dataframe.iloc[:, :-1].values
labels = dataframe.iloc[:, -1].values

# 归一化特征数据（如果需要）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# 转换为PyTorch张量
features_tensor = torch.tensor(features, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.float32)


# 创建数据集和数据加载器
class SurveyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


dataset = SurveyDataset(features_tensor, labels_tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义设备
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# 初始化网络
input_size = features_tensor.size(1)  # 特征数量
model = DepressionNet(input_size=input_size, hidden_sizes=[64, 32], output_size=1)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()  # 确保标签是浮点数

        optimizer.zero_grad()
        outputs = model(inputs)

        # 使用sigmoid激活函数将模型输出转换为概率值
        probabilities = torch.sigmoid(outputs.squeeze())

        loss = criterion(probabilities, labels)  # 现在使用概率值计算损失
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 预测
model.eval()
with torch.no_grad():
    predictions = model(features_tensor.to(device))

# 将预测结果转换为标签（0或1）
predicted_labels = (predictions > 0.5).cpu().numpy()

# 将标签转换为numpy数组以进行准确率计算
labels_np = labels_tensor.cpu().numpy()

# 计算准确率
accuracy = (predicted_labels == labels_np).mean()
print(f'Accuracy: {accuracy:.4f}')

# 输出前10个预测值和真实值
for i in range(10):
    print(f"Predicted: {predicted_labels[i]}, True Value: {labels_np[i]}")
