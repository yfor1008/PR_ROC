#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : pr_roc_test.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2021/08/28 16:21:43
# @Docs   : PR 及 ROC 测试, 动态图显示
'''

import os
from sklearn import svm
import numpy as np
import pickle

from sklearn.datasets._samples_generator import make_classification
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def get_PR_ROC(y_true, y_score):
    '''
    ### Docs: 获取二分类 PR 和 ROC
    ### Args:
        - y_true: array, N*1, 标签, [0,1]
        - y_score: array, N*1, 预测概率
    ### Returns:
        - PRs: (precision, recall, thres)
        - ROCs: (fpr, tpr, thres)
    '''

    # 按预测概率(score)降序排列
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # 概率(score)阈值, 取所有概率中不相同的
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size-1]
    thresholds = y_score[threshold_idxs]

    # 累计求和, 得到不同阈值下的 tps, fps
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # PR
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    PRs = (np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl])

    # ROC
    tps = np.r_[tps[sl], 0]
    fps = np.r_[fps[sl], 0]
    fpr = fps / fps[0]
    tpr = tps / tps[0]
    ROCs = (fpr, tpr, thresholds[sl])

    return PRs, ROCs

if __name__ == '__main__':

    # 生成测试数据: 前200训练, 后200测试
    if not os.path.exists('./src/random_test.npz'):
        X, y = make_classification(n_samples=20000, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0, n_informative=2)
        np.savez('./src/random_test', X=X, y=y)
    else:
        data = np.load('./src/random_test.npz')
        X = data['X']
        y = data['y']
    
    # 区分类别
    X0 = X[y==0, :]
    X1 = X[y==1, :]
    y0 = y[y==0]
    y1 = y[y==1]
    
    train_num = 1000
    test_num = 100
    X_train = np.vstack((X0[:train_num, :], X1[:train_num, :]))
    y_train = np.hstack((y0[:train_num], y1[:train_num]))
    X_test = np.vstack((X0[train_num:train_num+test_num, :], X1[train_num:train_num+test_num, :]))
    y_test = np.hstack((y0[train_num:train_num+test_num], y1[train_num:train_num+test_num]))

    # # 显示训练数据
    # unique_lables = set(y)
    # colors=plt.cm.Spectral(np.linspace(0,1,len(unique_lables)))
    # for k, col in zip(unique_lables, colors):
    #     x_k = X[y==k]
    #     plt.plot(x_k[:,0], x_k[:,1], 'o', markerfacecolor=col, markeredgecolor="k", markersize=5)
    # plt.title('data by make_classification()')
    # plt.show()

    # 模型测试
    model_file = './src/model'
    if not os.path.exists(model_file):
        random_state = np.random.RandomState(0)
        classifier = svm.SVC(random_state=random_state, probability=True)
        classifier.fit(X_train, y_train)
        with open(model_file, 'wb') as fw:
            pickle.dump(classifier, fw)
    else:
        with open(model_file, 'rb') as fr:
            classifier = pickle.load(fr)
    y_score = classifier.predict_proba(X_test)
    # print(y_score)
    # print(y_train)

    [prs, rocs] = get_PR_ROC(y_test, y_score[:, 1])

    # 显示PR, ROC

    fig = plt.figure()

    plt.subplot(3,1,1)
    pr_ani = plt.plot(prs[1], prs[0])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    pr_marker_ani = plt.plot(prs[1][0], prs[0][0], marker='o')[0]

    plt.subplot(3,1,2)
    plt.plot(rocs[0], rocs[1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    roc_marker_ani = plt.plot(rocs[0][0], rocs[1][0], marker='o')[0]

    plt.subplot(3,1,3)
    hist0 = plt.hist(y_score[y_test == 0, 1], bins=np.linspace(0,1,11), alpha=0.6, rwidth=0.6)
    hist1 = plt.hist(y_score[y_test == 1, 1], bins=np.linspace(0,1,11), alpha=0.6, rwidth=0.6)
    plt.legend(['Negative', 'Positive'], loc='upper center')
    max_num = max(max(hist0[0]), max(hist1[0]))
    thres_ani = plt.plot([prs[2][0]] * 10, np.linspace(0, max_num, 10))[0]
    plt.xlabel('score')
    plt.ylabel('sample num')

    plt.tight_layout()

    def update_pr(idx):
        pr_marker_ani.set_data(prs[1][idx], prs[0][idx])
        roc_marker_ani.set_data(rocs[0][idx], rocs[1][idx])
        thres_ani.set_data([prs[2][idx]] * 10, np.linspace(0, max_num, 10))

    ani = animation.FuncAnimation(fig=fig, func=update_pr, frames=np.arange(0, len(prs[2])), interval=50)
    ani.save('./src/pr_roc.gif', dpi=100)
    plt.show()
