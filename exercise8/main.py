# 8. 练习：使用特征预处理提升糖尿病患病指标预测模型的性能
# * 题目描述：对特征进行预处理，然后预测糖尿病的患病指标，并比较模型的性能 

# * 题目要求: 
# * 对类别型特征及数值型特征进行预处理 

# * 数据文件： 
# * 数据源下载地址：https://video.mugglecode.com/diabetes.csv (数据源与之前相同) 
# * diabetes.csv，包含了442个数据样本。 
# * 共11列数据 
# * AGE：年龄 
# * SEX: 性别 
# * BMI: 体质指数（Body Mass Index） 
# * BP: 平均血压（Average Blood Pressure） 
# * S1~S6: 一年后的6项疾病级数指标 
# * Y: 一年后患疾病的定量指标，为需要预测的标签


# * 问题拆解提示：
# 1. 如何处理类别型特征？
# 2. 如何处理数值型特征？
# 3. 如何合并预处理后的特征？
# * 问题解决提示：
# 1. 对类别型特征使用scikit-learn提供的独热编码OneHotEncoder()(http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)进行预处理；
# 2. 对数值型特征使用scikit-learn提供的最大最小归一化MinMaxScaler()(http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)进行预处理；
# 3. 使用NumPy提供的hstack()(https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html)对预处理后的特征在水平方向上进行合并


# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np


# 使用的特征列
NUMERIC_FEAT_COLS = ['AGE', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']
CATEGORY_FEAT_COLS = ['SEX']


def process_features(X_train, X_test):
    """
        特征预处理
    """
    # 1. 对类别型特征做one-hot encoding
    encoder = OneHotEncoder(sparse=False)
    encoded_tr_feat = encoder.fit_transform(X_train[CATEGORY_FEAT_COLS])
    encoded_te_feat = encoder.transform(X_test[CATEGORY_FEAT_COLS])

    # 2. 对数值型特征值做归一化处理
    scaler = MinMaxScaler()
    scaled_tr_feat = scaler.fit_transform(X_train[NUMERIC_FEAT_COLS])
    scaled_te_feat = scaler.transform(X_test[NUMERIC_FEAT_COLS])

    # 3. 特征合并
    X_train_proc = np.hstack((encoded_tr_feat, scaled_tr_feat))
    X_test_proc = np.hstack((encoded_te_feat, scaled_te_feat))

    return X_train_proc, X_test_proc


def main():
    """
        主函数
    """
    diabetes_data = pd.read_csv('./diabetes.csv')

    X = diabetes_data[NUMERIC_FEAT_COLS + CATEGORY_FEAT_COLS]
    y = diabetes_data['Y']

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=10)

    # 建立线性回归模型
    linear_reg_model = LinearRegression()
    # 模型训练
    linear_reg_model.fit(X_train, y_train)
    # 验证模型
    r2_score = linear_reg_model.score(X_test, y_test)
    print('模型的R2值', r2_score)

    # 数据预处理
    X_train_proc, X_test_proc = process_features(X_train, X_test)
    # 建立线性回归模型
    linear_reg_model2 = LinearRegression()
    # 模型训练
    linear_reg_model2.fit(X_train_proc, y_train)
    # 验证模型
    r2_score2 = linear_reg_model2.score(X_test_proc, y_test)
    print('特征处理后，模型的R2值', r2_score2)

    print('模型提升了{:.2f}%'.format((r2_score2 - r2_score) / r2_score * 100))


if __name__ == '__main__':
    main()
    