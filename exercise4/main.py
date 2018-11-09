# 4. 练习：预测糖尿病的患病指标
# * 题目描述：预测糖尿病的患病指标 

# * 题目要求: 
# * 使用scikit-learn的线型回归模型对糖尿病的指标值进行预测 
# * 选取1/5的数据作为测试集 

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



# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 使用的特征列
FEAT_COLS = ['AGE', 'SEX', 'BMI', 'BP', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6']


def main():
    """
        主函数
    """
    diabetes_data = pd.read_csv('./diabetes.csv')

    X = diabetes_data[FEAT_COLS].values
    y = diabetes_data['Y'].values

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/5, random_state=10)

    # 1. 建立线性回归模型
    linear_reg_model = LinearRegression()
    # 2. 模型训练
    linear_reg_model.fit(X_train, y_train)
    # 3. 验证模型
    r2_score = linear_reg_model.score(X_test, y_test)
    print('模型的R2值', r2_score)


if __name__ == '__main__':
    main()