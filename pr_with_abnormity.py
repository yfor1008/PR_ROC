#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @File   : pr_with_abnormity.py
# @Author : yuanwenjin
# @Mail   : xxxx@mail.com
# @Date   : 2021/08/28 16:49:56
# @Docs   : 存在问题的PR曲线
'''

import numpy as np
from sklearn.metrics  import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def test_case1():
    '''概率很高负样本示例'''
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
    label = np.array([0, 1, 1, 1, 0, 0, 0])

    precision, recall, thres = precision_recall_curve(label, score)
    thres = np.r_[thres, thres[-1]]

    fig = plt.figure()
    plt.plot(recall, precision)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig('./src/pr_high_porb_negative.png')
    plt.show()

    fig,ax = plt.subplots()

    idx = 0
    plt.plot(recall, precision)
    plt.plot(recall, thres)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(['PR', 'thres'])
    pr_ani = plt.plot(recall[idx], precision[idx], marker='o')[0]
    thre_ani = plt.plot(recall[idx], thres[idx], marker='o')[0]

    plt.tight_layout()

    def update_pr(idx):

        pr_ani.set_data(recall[idx], precision[idx])
        thre_ani.set_data(recall[idx], thres[idx])

    ani = animation.FuncAnimation(fig=fig, func=update_pr, frames=np.arange(0, len(precision)), interval=500)
    ani.save('./src/pr_thres.gif', dpi=100)
    plt.show()

def test_case2():
    '''概率很低正样本示例'''
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
    label = np.array([1, 1, 1, 0, 0, 0, 1])

    precision, recall, thres = precision_recall_curve(label, score)
    thres = np.r_[thres, thres[-1]]

    fig = plt.figure()
    plt.plot(recall, precision)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig('./src/pr_low_porb_positive.png')
    plt.show()

if __name__=='__main__':

    # case 1
    test_case1()

    # case 2
    test_case2()

