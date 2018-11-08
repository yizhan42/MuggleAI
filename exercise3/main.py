# * 题目描述：使用不同的k值，观察对水果识别器的影响。 

# * 题目要求: 
# * 使用scikit-learn的kNN进行识别 
# * 使用k=1, 3, 5, 7观察对结果的影响 

# * 数据文件： 
# * 数据源下载地址：https://video.mugglecode.com/fruit_data.csv（数据源与上节课相同） 
# * fruit_data.csv，包含了60个水果的的数据样本。 
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
# 1. 如何指定k值？
# * 问题解决提示：
# 1. 利用scikit-learn模块中的KNeighborsClassifier()(http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)建立kNN模型，其中参数n_neighbors是kNN中的k值，默认值为5


# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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

    # 1. 处理样本的字符串标签，添加label一列作为预测标签
    fruit_data['label'] = fruit_data['fruit_name'].map(CATEGRORY_LABEL_DICT)

    # 4列水果的属性作为样本特征
    X = fruit_data[FEAT_COLS].values
    # label列为样本标签
    y = fruit_data['label'].values

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=20)

    print('原始数据集共{}个样本，其中训练集样本数为{}，测试集样本数为{}'.format(
        X.shape[0], X_train.shape[0], X_test.shape[0]))

    # 2. 建立kNN模型
    knn_model = KNeighborsClassifier()
    # 3. 训练模型
    knn_model.fit(X_train, y_train)

    # 4. 验证模型
    accuracy = knn_model.score(X_test, y_test)
    print('预测准确率为：{:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()