# -*- coding: utf-8 -*-
# 所需数据集请下载：https://video.mugglecode.com/Iris.csv

"""
    任务：鸢尾花识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = './data/Iris.csv'

CATEGRORY_LABEL_DICT = {
        'Iris-setosa':      0,  # 山鸢尾
        'Iris-versicolor':  1,  # 变色鸢尾
        'Iris-virginica':   2   # 维吉尼亚鸢尾
    }

# 使用的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def main():
    """
        主函数
    """
    iris_data = pd.read_csv(DATA_FILE, index_col='Id')

    # 添加label一列作为预测标签
    iris_data['Label'] = iris_data['Species'].apply(lambda category_name: CATEGRORY_LABEL_DICT[category_name])

    # 4列花的属性作为样本特征
    X = iris_data[FEAT_COLS].values
    # label列为样本标签
    y = iris_data['Label'].values

    # 将原始数据集拆分成训练集和测试集，测试集占总样本数的1/3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    # 构建模型
    max_depth_list = [2, 3, 4]

    for max_depth in max_depth_list:
        dt_model = DecisionTreeClassifier(max_depth=max_depth)
        dt_model.fit(X_train, y_train)

        train_acc = dt_model.score(X_train, y_train)
        test_acc = dt_model.score(X_test, y_test)

        print('max_depth', max_depth)
        print('训练集上的准确率：{:.2f}%'.format(train_acc * 100))
        print('测试集上的准确率：{:.2f}%'.format(test_acc * 100))
        print()


if __name__ == '__main__':
    main()