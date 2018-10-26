# -*- coding: utf-8 -*-
# 所需数据集请下载：https://video.mugglecode.com/Iris.csv

"""
    任务：鸢尾花识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import matplotlib.pyplot as plt
import numpy as np

DATA_FILE = './data/Iris.csv'

CATEGRORY_LABEL_DICT = {
        'Iris-setosa':      0,  # 山鸢尾
        'Iris-versicolor':  1,  # 变色鸢尾
        'Iris-virginica':   2   # 维吉尼亚鸢尾
    }

# 使用的特征列
FEAT_COLS = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


def plot_decision_tree(dt_model):
    """
        可视化树的结构
    """
    tmp_dot_file = 'decision_tree_tmp.dot'
    cat_names = list(CATEGRORY_LABEL_DICT.keys())
    export_graphviz(dt_model, out_file=tmp_dot_file, feature_names=FEAT_COLS, class_names=cat_names,
                    filled=True, impurity=False)
    with open(tmp_dot_file) as f:
        dot_graph = f.read()
    graph = pydotplus.graph_from_dot_data(dot_graph)
    graph.write_png('decision_tree.png')


def inspect_feature_importances(dt_model):
    """
        特征重要性
    """
    print('特征名称：', FEAT_COLS)
    print('特征重要性：', dt_model.feature_importances_)
    n_features = len(FEAT_COLS)

    plt.figure()
    plt.barh(range(n_features), dt_model.feature_importances_)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature name')
    plt.yticks(np.arange(n_features), FEAT_COLS)
    plt.tight_layout()
    plt.show()


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

    # 建立模型
    dt_model = DecisionTreeClassifier(max_depth=4)
    dt_model.fit(X_train, y_train)

    # 可视化树的结构
    plot_decision_tree(dt_model)

    # 特征重要性
    inspect_feature_importances(dt_model)


if __name__ == '__main__':
    main()