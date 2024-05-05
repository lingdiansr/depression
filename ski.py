import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 读取CSV文件
df = pd.read_csv('dataset/b_depressed.csv')

# 数据预处理（例如，将分类变量转换为独热编码）
# 假设'sex'是分类变量，需要转换
df = pd.get_dummies(df, columns=['sex'], drop_first=True)

# 假设最后一列'depressed'是目标变量，其余列是特征
X = df.drop('depressed', axis=1)  # 特征
y = df['depressed']  # 目标

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 创建BP神经网络模型
mlp = MLPClassifier(hidden_layer_sizes=(100,), random_state=42, max_iter=500)

# 训练模型
mlp.fit(X_train, y_train)

# 预测测试集
y_pred = mlp.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# 保存模型，以便将来使用
mlp.save('bp_neural_network.joblib')