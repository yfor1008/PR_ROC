#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : pr_roc_samples.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2021/08/28 16:21:43
# @Docs   : 样本不均衡测试
'''

import os
from types import prepare_class
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
    precision[np.isnan(precision)] = 0 # 将nan替换为0
    recall = tps / tps[-1]
    last_ind = tps.searchsorted(tps[-1]) # 最后一个tps的index
    sl = slice(last_ind, None, -1) # 倒序
    precision = np.r_[precision[sl], 1] # 添加 precision=1, recall=0, 可以让数据从0开始
    recall = np.r_[recall[sl], 0]
    PRs = (precision, recall, thresholds[sl])

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
    
    # 不同比例样本测试
    train_num = 1000
    test_num = 100
    X_train = np.vstack((X0[:train_num, :], X1[:train_num, :]))
    y_train = np.hstack((y0[:train_num], y1[:train_num]))
    prs_list = []
    rocs_list = []
    hist0_list = []
    hist1_list = []
    for idx in range(1, 101, 10):
        X_test = np.vstack((X0[train_num:train_num+test_num*idx, :], X1[train_num:train_num+test_num, :]))
        y_test = np.hstack((y0[train_num:train_num+test_num*idx], y1[train_num:train_num+test_num]))

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

        [prs, rocs] = get_PR_ROC(y_test, y_score[:, 1])
        prs_list.append(prs)
        rocs_list.append(rocs)

        hist0, bins = np.histogram(y_score[:, 1][y_test == 0], range=(0,1), bins=10)
        hist1, _ = np.histogram(y_score[:, 1][y_test == 1], range=(0,1), bins=10)
        hist0_list.append(hist0)
        hist1_list.append(hist1)

    # 显示PR, ROC
    fig = plt.figure()

    def update_pr(idx):

        ax1 = plt.subplot(3,1,1)
        ax1.cla()
        plt.plot(prs_list[idx][1], prs_list[idx][0])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        ax2 = plt.subplot(3,1,2)
        ax2.cla()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot(rocs_list[idx][0], rocs_list[idx][1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

        ax3 = plt.subplot(3,1,3)
        ax3.cla()
        plt.bar(bins[1:], hist0_list[idx], alpha=0.6, width=0.1)
        plt.bar(bins[1:], hist1_list[idx], alpha=0.6, width=0.1)
        plt.legend(['Negative: %d' % np.sum(hist0_list[idx]), 'Positive: %d' % np.sum(hist1_list[idx])], loc='upper center')
        plt.xlabel('score')
        plt.ylabel('sample num')

        plt.tight_layout()

    ani = animation.FuncAnimation(fig=fig, func=update_pr, frames=np.arange(0, len(hist0_list)), interval=500)
    ani.save('./src/pr_roc_samples.gif', dpi=100)
    plt.show()
