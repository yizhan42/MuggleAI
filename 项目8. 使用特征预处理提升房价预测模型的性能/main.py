# 人工智能数据源下载地址：https://video.mugglecode.com/data_ai.zip，下载压缩包后解压即可（数据源与上节课相同）
# -*- coding: utf-8 -*-

"""
    任务：房屋价格预测
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

DATA_FILE = './data_ai/house_data.csv'

# 使用的特征列
NUMERIC_FEAT_COLS = ['sqft_living', 'sqft_above', 'sqft_basement', 'long', 'lat']
CATEGORY_FEAT_COLS = ['waterfront']


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
    house_data = pd.read_csv(DATA_FILE, usecols=NUMERIC_FEAT_COLS + CATEGORY_FEAT_COLS + ['price'])

    X = house_data[NUMERIC_FEAT_COLS + CATEGORY_FEAT_COLS]
    y = house_data['price']

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

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

    print('模型提升了{:.2f}%'.format( (r2_score2 - r2_score) / r2_score * 100) )


if __name__ == '__main__':
    main()