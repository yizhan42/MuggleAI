# 1. 练习：手工实现一个简单的水果识别器
# * 题目描述：创建一个水果识别器，根据水果的属性，判断该水果的种类。 

# * 题目要求: 
# * 模仿课堂的讲解内容，根据“近朱者赤”的原则，手工实现一个简单的分类器 
# * 选取1/5的数据作为测试集 

# * 数据文件： 
# * 数据源下载地址：https://video.mugglecode.com/fruit_data.csv 
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
# * 如图所示:https://video.mugglecode.com/color_score.jpg
# -*- coding: utf-8 -*-

# 1. 练习：手工实现一个简单的水果识别器
# * 题目描述：创建一个水果识别器，根据水果的属性，判断该水果的种类。

# * 题目要求:
# * 模仿课堂的讲解内容，根据“近朱者赤”的原则，手工实现一个简单的分类器
# * 选取1/5的数据作为测试集

# * 数据文件：
# * 数据源下载地址：https://video.mugglecode.com/fruit_data.csv
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
# * 如图所示:https://video.mugglecode.com/color_score.jpg
# -*- coding: utf-8 -*-

"""
    任务：水果识别
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np

# import ai_utils

DATA_FILE = './fruit_data.csv'

SPECIES = ['apple',
           'mandarin',
           'orange',
           'lemon'
           ]

# 使用的特征列
FEAT_COLS = ['mass', 'width', 'height', 'color-score']


def get_pred_label(test_sample_feat, train_data):
    """
        “近朱者赤” 找最近距离的训练样本，取其标签作为预测样本的标签
    """
    dis_list = []

    for idx, row in train_data.iterrows():
        # 训练样本特征
        train_sample_feat = row[FEAT_COLS].values

        # 计算距离
        dis = euclidean(test_sample_feat, train_sample_feat)
        dis_list.append(dis)

    # 最小距离对应的位置
    pos = np.argmin(dis_list)
    pred_label = train_data.iloc[pos]['fruit_name']
    return pred_label


def main():
    """
        主函数
    """
    # 读取数据集
    fruit_data = pd.read_csv(DATA_FILE)

    # EDA
    # ai_utils.do_eda_plot_for_iris(iris_data)

    # 划分数据集
    train_data, test_data = train_test_split(fruit_data, test_size=1/5, random_state=10)

    # 预测对的个数
    acc_count = 0

    # 分类器
    for idx, row in test_data.iterrows():
        # 测试样本特征
        test_sample_feat = row[FEAT_COLS].values

        # 预测值
        pred_label = get_pred_label(test_sample_feat, train_data)

        # 真实值
        true_label = row['fruit_name']
        print('样本{}的真实标签{}，预测标签{}'.format(idx, true_label, pred_label))

        if true_label == pred_label:
            acc_count += 1

    # 准确率
    accuracy = acc_count / test_data.shape[0]
    print('预测准确率{:.2f}%'.format(accuracy * 100))


if __name__ == '__main__':
    main()