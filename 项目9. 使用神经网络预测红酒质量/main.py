# -*- coding: utf-8 -*-
# 所需数据请在这里下载：https://video.mugglecode.com/wine_quality.csv

"""
    任务：红酒质量预测
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

DATA_FILE = './data/wine_quality.csv'


def main():
    """
        主函数
    """
    wine_data = pd.read_csv(DATA_FILE)
    # sns.countplot(data=wine_data, x='quality')
    # plt.show()

    # 数据预处理
    wine_data.loc[wine_data['quality'] <= 5, 'quality'] = 0
    wine_data.loc[wine_data['quality'] >= 1, 'quality'] = 1

    # sns.countplot(data=wine_data, x='quality')
    # plt.show()

    # 所有列名
    all_cols = wine_data.columns.tolist()

    # 特征列名称
    feat_cols = all_cols[:-1]

    # 特征
    X = wine_data[feat_cols].values
    # 标签
    y = wine_data['quality'].values

    # 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    # 特征归一化
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 建立模型
    mlp = MLPClassifier(hidden_layer_sizes=(50, 50, 100), activation='relu')
    mlp.fit(X_train_scaled, y_train)
    accuracy = mlp.score(X_test_scaled, y_test)
    print('神经网络模型的预测准确率：{:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()