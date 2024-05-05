import pandas as pd
from sklearn.model_selection import train_test_split

# 假设数据集已经被加载到DataFrame中，这里命名为df
dataset_path = './dataset/'
df = pd.read_csv(dataset_path+'b_depressed.csv')

# 随机挑选300条数据作为测试集，其余作为训练集
train_df, test_df = train_test_split(df, test_size=300, random_state=42)

# 将训练集和测试集保存为CSV文件
train_df.to_csv(dataset_path+'train.csv', index=False)
test_df.to_csv(dataset_path+'test.csv', index=False)