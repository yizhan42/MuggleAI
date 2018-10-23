# 人工智能数据源下载地址：https://video.mugglecode.com/data_ai.zip，下载压缩包后解压即可（数据源与上节课相同）
# -*- coding: utf-8 -*-

"""
    任务：房屋价格预测
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

DATA_FILE = './data_ai/house_data.csv'

# 使用的特征列
FEAT_COLS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement']


def plot_fitting_line(linear_reg_model, X, y, feat):
    """
        绘制线型回归线
    """
    w = linear_reg_model.coef_
    b = linear_reg_model.intercept_

    plt.figure()
    # 样本点
    plt.scatter(X, y, alpha=0.5)

    # 直线
    plt.plot(X, w * X + b, c='red')
    plt.title(feat)
    plt.show()


def main():
    """
        主函数
    """
    house_data = pd.read_csv(DATA_FILE, usecols=FEAT_COLS + ['price'])

    for feat in FEAT_COLS:
        X = house_data[feat].values.reshape(-1, 1)
        y = house_data['price'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train, y_train)
        r2_score = linear_reg_model.score(X_test, y_test)
        print('特征：{}，R2值：{}'.format(feat, r2_score))

        # 绘制拟合直线
        plot_fitting_line(linear_reg_model, X_test, y_test, feat)


if __name__ == '__main__':
    main()