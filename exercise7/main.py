# 7. 练习：使用交叉验证对水果分类模型进行调参
# * 题目描述：为模型选择最优的参数并进行水果类型识别，模型包括kNN，逻辑回归及SVM。对应的超参数为： 
# * kNN中的近邻个数n_neighbors及闵式距离的p值 
# * 逻辑回归的正则项系数C值 
# * SVM的正则项系数C值 

# * 题目要求: 
# * 使用3折交叉验证对模型进行调参 
# * 使用scikit-learn提供的方法为模型调参 

# * 数据文件： 
# * 数据源下载地址：https://video.mugglecode.com/fruit_data.csv（数据源与上节课相同） 
# * fruit_data.csv，包含了59个水果的的数据样本。 
# * 共5列数据 
# * fruit_name：水果类别 
# * mass: 水果质量 
# * width: 水果的宽度 
# * height: 水果的高度 
# * color_score: 水果的颜色数值，范围0-1。 
# * 0.85 - 1.00：红色 
# * 0.75 - 0.85: 橙色 
# * 0.65 - 0.75: 黄色 
# * 0.45 - 0.65: 绿色 
# * 如图所示：https://video.mugglecode.com/color_score.jpg

# * 问题拆解提示：
# 1. 如何为模型选择最优的超参数？
# * 问题解决提示：
# 1. 使用scikit-learn中的网格搜索GridSearchCV()(http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)为模型选择最优的超参数，常用的参数包括：
# * estimator：需要调参的模型
# * param_grid：参数空间，以字典的形式给出
# * cv：交叉验证的折数，这里cv=3

# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

CATEGRORY_LABEL_DICT = {
    'apple':    0,
    'lemon':    1,
    'mandarin': 2,
    'orange':   3
}

# 使用的特征列
FEAT_COLS = ['mass', 'width', 'height', 'color_score']


def main():
    """
        主函数
    """
    fruit_data = pd.read_csv('./fruit_data.csv')

    # 处理样本的字符串标签，添加label一列作为预测标签
    fruit_data['label'] = fruit_data['fruit_name'].map(CATEGRORY_LABEL_DICT)

    # 4列水果的属性作为样本特征
    X = fruit_data[FEAT_COLS].values
    # label列为样本标签
    y = fruit_data['label'].values

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=20)

    # 建立kNN、逻辑回归和SVM模型
    model_dict = {'kNN': (KNeighborsClassifier(),
                          {'n_neighbors': [5, 15, 25],
                           'p': [1, 2]}),
                  'Logistic Regression': (LogisticRegression(),
                                          {'C': [1e-2, 1, 1e2]}),
                  'SVM': (SVC(),
                          {'C': [1e-2, 1, 1e2]})
                  }

    for model_name, (model, model_params) in model_dict.items():
        # 1. 使用网格搜索为模型选择最优的超参数
        clf = GridSearchCV(estimator=model, param_grid=model_params, cv=3)
        clf.fit(X_train, y_train)
        best_model = clf.best_estimator_

        # 验证
        acc = best_model.score(X_test, y_test)
        print('{}模型的预测准确率：{:.2f}%'.format(model_name, acc * 100))
        print('{}模型的最优参数：{}'.format(model_name, clf.best_params_))


if __name__ == '__main__':
    main()