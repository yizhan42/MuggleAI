# -*- coding: utf-8 -*-
# 所需数据请在这里下载：https://video.mugglecode.com/wine_quality.csv

"""
    任务：红酒质量预测
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


DATA_FILE = './data/wine_quality.csv'


def main():
    """
        主函数
    """
    wine_data = pd.read_csv(DATA_FILE)
    # 处理数据
    wine_data.loc[wine_data['quality'] <= 5, 'quality'] = 0
    wine_data.loc[wine_data['quality'] >= 6, 'quality'] = 1
    all_cols = wine_data.columns.tolist()
    feat_cols = all_cols[:-1]

    # 11列红酒的属性作为样本特征
    X = wine_data[feat_cols].values
    # label列为样本标签
    y = wine_data['quality'].values

    # 将原始数据集拆分成训练集和测试集，测试集占总样本数的1/3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=10)

    # 特征预处理
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 构建组件分类器
    clf1 = DecisionTreeClassifier(max_depth=10)
    clf2 = LogisticRegression(C=0.1)
    clf3 = SVC(kernel='linear', probability=True)

    clfs = [('决策树', clf1), ('逻辑回归', clf2), ('支持向量机', clf3)]

    for clf_tup in clfs:
        clf_name, clf = clf_tup
        clf.fit(X_train_scaled, y_train)
        acc = clf.score(X_test_scaled, y_test)
        print('模型：{}, 准确率:{:.2f}%'.format(clf_name, acc * 100))

    # hard voting
    hard_clf = VotingClassifier(estimators=clfs, voting='hard')
    hard_clf.fit(X_train_scaled, y_train)
    print('hard voting: {:.2f}%'.format(hard_clf.score(X_test_scaled, y_test) * 100))

    # soft voting
    soft_clf = VotingClassifier(estimators=clfs, voting='soft')
    soft_clf.fit(X_train_scaled, y_train)
    print('soft voting: {:.2f}%'.format(soft_clf.score(X_test_scaled, y_test) * 100))


if __name__ == '__main__':
    main()