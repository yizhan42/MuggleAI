# 人工智能数据源下载地址：https://video.mugglecode.com/data_ai.zip，下载压缩包后解压即可（数据源与上节课相同）
# -*- coding: utf-8 -*-

"""
    任务：鸢尾花识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import ai_utils

DATA_FILE = './data_ai/Iris.csv'

SPECIES_LABEL_DICT = {
    'Iris-setosa':      0,  # 山鸢尾
    'Iris-versicolor':  1,  # 变色鸢尾
    'Iris-virginica':   2   # 维吉尼亚鸢尾
}

# 使用的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def investigate_knn(iris_data, sel_cols, k_val):
    """
        不同的K值对模型的影响
    """
    X = iris_data[sel_cols].values
    y = iris_data['Label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)
    knn_model = KNeighborsClassifier(n_neighbors=k_val)
    knn_model.fit(X_train, y_train)
    accuracy = knn_model.score(X_test, y_test)
    print('k={}，accuracy={:.2f}%'.format(k_val, accuracy * 100))

    ai_utils.plot_knn_boundary(knn_model, X_test, y_test,
                               'Sepal Length vs Sepal Width, k={}'.format(k_val),
                               save_fig='sepal_k={}.png'.format(k_val))


def main():
    """
        主函数
    """
    # 读取数据集
    iris_data = pd.read_csv(DATA_FILE, index_col='Id')
    iris_data['Label'] = iris_data['Species'].map(SPECIES_LABEL_DICT)

    k_vals = [3, 5, 10]
    sel_cols = ['SepalLengthCm', 'SepalWidthCm']
    for k_val in k_vals:
        investigate_knn(iris_data, sel_cols, k_val)


if __name__ == '__main__':
    main()