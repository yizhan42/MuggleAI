# 人工智能数据源下载地址：https://video.mugglecode.com/data_ai.zip，下载压缩包后解压即可（数据源与上节课相同）
# -*- coding: utf-8 -*-

"""
    任务：鸢尾花识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


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

    model_dict = {'kNN': (KNeighborsClassifier(),
                          {'n_neighbors': [5, 15, 25],
                           'p': [1, 2]}),
                  'Logistic Regression': (LogisticRegression(),
                                          {'C': [1e-2, 1, 1e2]}),
                  'SVM': (SVC(),
                          {'C': [1e-2, 1, 1e2]})
                  }

    for model_name, (model, model_params) in model_dict.items():
        # 训练模型
        clf = GridSearchCV(estimator=model, param_grid=model_params, cv=5)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        # 验证
        acc = best_model.score(X_test, y_test)
        print('{}模型的预测准确率：{:.2f}%'.format(model_name, acc * 100))
        print('{}模型的最优参数：{}'.format(model_name, clf.best_params_))


if __name__ == '__main__':
    main()