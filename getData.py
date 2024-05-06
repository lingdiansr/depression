import pandas as pd
from sklearn.model_selection import train_test_split

class DataSet():
    def __init__(self):
        # 加载数据集
        self.dataset_path = './dataset/b_depressed.csv'
        data = pd.read_csv(self.dataset_path)

        # 随机选择20%的数据作为测试集，80%作为训练集
        test_data, train_data = train_test_split(data, test_size=0.2, random_state=42)

        # 将测试集和训练集分别保存为CSV文件
        self.test_dataset_path = './dataset/test.csv'
        self.train_dataset_path = './dataset/train.csv'

        test_data.to_csv(self.test_dataset_path, index=False)
        train_data.to_csv(self.train_dataset_path, index=False)

        print(f"Test set saved to {self.test_dataset_path}")
        print(f"Train set saved to {self.train_dataset_path}")
    def get_trainset(self):
        return self.train_dataset_path
    def get_testset(self):
        return self.test_dataset_path