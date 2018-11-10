# 6. 练习：使用kNN、逻辑回归和SVM进行水果类型识别
# * 题目描述：使用kNN、逻辑回归和SVM进行水果类型识别 

# * 题目要求: 
# * 使用scikit-learn提供的kNN、逻辑回归和SVM进行分类操作 
# * 手动选择合适的模型超参数，包括kNN中的近邻个数k，逻辑回归和SVM中的正则项系数C值 

# * 数据文件： 
# * 数据源下载地址：https://video.mugglecode.com/fruit_data.csv（数据源与之前相同） 
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
# 1. 如何建立逻辑回归和SVM模型？
# 2. 如何为模型设置超参数？
# * 问题解决提示：
# 1. 使用scikit-learn中的LogisticRegression()(http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)和SVC()(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)分别建立逻辑回归和SVM模型；
# 2. 通过KNeighborsClassifier()(http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)中的参数n_neighbors设置k值，通过LogisticRegression(http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)和SVC()(http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)中的参数C设置正则项系数C（本练习中只需要手动设置，无需使用“交叉验证”）




# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
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

    # 1. 建立kNN、逻辑回归和SVM模型
    # 2. 为模型分别设置超参数
    model_dict = {'kNN': KNeighborsClassifier(n_neighbors=3),
                  'Logistic Regression': LogisticRegression(C=1e3),
                  'SVM': SVC(C=1e3)}

    for model_name, model in model_dict.items():
        # 训练模型
        model.fit(X_train, y_train)
        # 验证模型
        acc = model.score(X_test, y_test)
        print('{}模型的预测准确率:{:.2f}%'.format(model_name, acc * 100))


if __name__ == '__main__':
    main()