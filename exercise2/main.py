# 2. 练习：使用 scikit-learn 的 kNN 分类算法实现水果识别器
# * 题目描述：使用k近邻距离算法创建一个水果识别器，根据水果的属性，判断该水果的种类。

# * 题目要求: 
# * 使用scikit-learn的kNN算法进行识别 

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
# 1. 如何处理样本的字符串标签？
# 2. 如何建立kNN模型？
# 3. 如何训练模型？
# 4. 如何验证模型？
# * 问题解决提示：
# 1. 利用Pandas模块中的map()(https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.map.html)方法进行字符串到数字的映射转换；
# 2. 利用scikit-learn模块中的KNeighborsClassifier()(http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)建立kNN模型；
# 3. 利用scikit-learn模块中的fit()(http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.fit)方法训练模型；
# 4. 利用scikit-learn模块中的score()(http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.score)方法验证模型。

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








