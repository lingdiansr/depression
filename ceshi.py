import torch
import numpy as np
import pandas as pd
import os
import depressionModule as module
import getData as DATA

# 确保模型文件存在
model_path = 'models/best_model.pth'
assert os.path.exists(model_path), "模型文件不存在，请检查路径是否正确。"

# 加载模型参数
model_state_dict = torch.load(model_path)
input_size = 22  # 与训练时的输入特征维度一致
hidden_sizes = [128, 100]  # 与训练时的隐藏层大小一致
output_size = 1  # 与训练时的输出类别数量一致

# 将模型参数加载到模型中
model = module.DepressionNet(input_size, hidden_sizes, output_size)
model.load_state_dict(model_state_dict)
model.eval()  # 设置为评估模式

# 确保模型使用的设备与训练时一致
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 获取测试数据集
# test_dataset = DATA.DataSet().get_testset()
data = pd.read_csv('dataset/test.csv')

# 提取特征和标签
X_test = data.iloc[:, :-1].values
y_test = data.iloc[:, -1].values

# 将测试数据转换为PyTorch张量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 使用训练好的模型进行预测
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, dim=1)  # 得到预测结果

# 将预测结果和真实标签转换为numpy数组进行比较
predicted_np = predicted.cpu().numpy()
y_test_np = y_test_tensor.cpu().numpy()

# 计算准确率
accuracy = np.mean(predicted_np == y_test_np)
print(f"Accuracy: {accuracy:.4f}")

# # 如果需要，可以输出每个预测结果与真实标签的比较
# for i, (pred, true) in enumerate(zip(predicted_np, y_test_np)):
#     print(f"[第{i+1}例，预测值：{pred}, 真实值：{true}]")