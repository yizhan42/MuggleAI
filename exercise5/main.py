# 5. 练习：糖尿病患病指标的可视化线性模型
# * 题目描述：可视化线型回归模型 

# * 题目要求: 
# * 绘制糖尿病患病的各项属性（特征）的线型拟合曲线 

# * 数据文件： 
# * 数据源下载地址：https://video.mugglecode.com/diabetes.csv 
# * diabetes.csv，包含了442个数据样本。 
# * 共11列数据 
# * AGE：年龄 
# * SEX: 性别 
# * BMI: 体质指数（Body Mass Index） 
# * BP: 平均血压（Average Blood Pressure） 
# * S1~S6: 一年后的6项疾病级数指标 
# * Y: 一年后患疾病的定量指标，为需要预测的标签


## 提示
# * 问题拆解提示：
# 1. 如何获取线型模型的权重和偏置项？
# 2. 如何绘制线型回归线？
# * 问题解决提示：
# 1. 线型回归模型LinearRegression()(http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)的参数coef_和intercept_为训练好的模型权重和偏置项；
# 2. 使用Matplotlib的scatter()(https://matplotlib.org/api/_as_gen/matplotlib.pyplot.scatter.html)方法绘制样本点，使用Matplotlib的plot()(https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)绘制直线。


# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# 使用的特征列
FEAT_COLS = ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']


def plot_fitting_line(linear_reg_model, X, y, feat):
    """
        绘制线型回归线
    """
    # 1. 获取线型模型的权重和偏置项
    w = linear_reg_model.coef_
    b = linear_reg_model.intercept_

    # 2. 绘制线型回归线
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
    diabetes_data = pd.read_csv('./diabetes.csv')

    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Y'].values

    for feat in FEAT_COLS:
        X = diabetes_data[feat].values.reshape(-1, 1)
        y = diabetes_data['Y'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=10)
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train, y_train)
        r2_score = linear_reg_model.score(X_test, y_test)
        print('特征：{}，R2值：{}'.format(feat, r2_score))

        # 绘制拟合直线
        plot_fitting_line(linear_reg_model, X_test, y_test, feat)


if __name__ == '__main__':
    main()