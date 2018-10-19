# 人工智能数据源下载地址：https://video.mugglecode.com/data_ai.zip，下载压缩包后解压即可（数据源与上节课相同）
# -*- coding: utf-8 -*-

"""
    任务：鸢尾花识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_FILE = './data_ai/Iris.csv'

SPECIES_LABEL_DICT = {
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
    # 读取数据集
    iris_data = pd.read_csv(DATA_FILE, index_col='Id')
    iris_data['Label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)

    # 获取数据集特征
    X = iris_data[FEAT_COLS].values

    # 获取数据标签
    y = iris_data['Label'].values

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    # 声明模型
    knn_model = KNeighborsClassifier()
    # 训练模型
    knn_model.fit(X_train, y_train)
    # 评价模型
    accuracy = knn_model.score(X_test, y_test)
    print('预测准确率:{:.2f}%'.format(accuracy * 100))

    # 取单个测试样本
    idx = 25
    test_sample_feat = [X_test[idx, :]]
    y_true = y_test[idx]
    y_pred = knn_model.predict(test_sample_feat)
    print('真实标签{}，预测标签{}'.format(y_true, y_pred))


if __name__ == '__main__':
    main()