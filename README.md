# PR 与 ROC
## PR(Precision Recall)曲线

### 问题

最近项目中遇到一个比较有意思的问题, 如下所示为:

![pr_high_porb_negative](https://gitee.com/yfor1008/pictures/raw/master/pr_high_porb_negative.png)

图中的`PR`曲线很奇怪, 左边从1突然变到0.

### PR源码分析

为了搞清楚这个问题, 对源码进行了分析. 如下所示为上图对应的代码:

```python
from sklearn.metrics  import precision_recall_curve
import matplotlib.pyplot as plt
score = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2, 0.1])
label = np.array([0, 1, 1, 1, 0, 0, 0])
precision, recall, thres = precision_recall_curve(label, score)
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
```

代码中得到`precision`和`recall`使用的是`sklearn.metrics.precision_recall_curve`, 下面为从其对应的[源码](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_ranking.py#L775)中抽取出来的关键代码:

```python
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
```

从代码中总结了计算`PR`的几个关键步骤:

1. 对于预测概率(score)排序, 从高到低
2. 以预测概率(score)作为阈值统计`tps`和`fps`
3. 计算`precision`和`recall`, 并倒序

这里补充说明几个特点:

1. 以测试数据的预测概率(score)作为阈值, 因而阈值只能在测试数据预测概率(score)集合中, 不是连续变化的;
2. 统计`tps`和`fps`时, 统计的是大于等于阈值的数据的个数, 因而理想情况下, `tps>=1`和`fps>=1`, 这里说的是理想情况下, 不理想情况后面说明;
3. 测试数据预测概率(score)可能不会出现为1的情况, 此种情况下, `recall=0`, 为了使得`PR`曲线从0开始, 添加了`recall=0, precision=1`;
4. 使用倒序, 让阈值从小到大, 因而`PR`曲线是从左向右画的, 如下图所示:

![](https://gitee.com/yfor1008/pictures/raw/master/pr_thres.gif)

### 问题原因分析

弄清楚了`PR`原理及计算方法, 就好分析上述问题产生的原因了.

#### 1的来历

从上述原理及计算过程分析可以看到, 最后添加了`recall=0, precision=1`, 对应图中最左边的1, 这里就知道了1是怎么来的;

#### 0的来历

`precision`的计算公式是`precision=TP/(TP+FP)`, 理想情况下(score值越大, Positive的可能性就越大), 随着阈值的增加, TP越来越小, FP越来越小, precision是越来越大的, 是不可能出现为0情况的; 只有当TP=0时, precision才会出现为0的情况, 这种情况属于非理想情况(score值越大, Positive的可能性不一定越大).

来看看`tps`的计算方法, 统计的是大于等于阈值thres的数据中为Positive的个数, 只有Positive个数为0的情况下, tps才能为0,  那么thres对应的数据就不是Positive的, 而是Negative的.

我们来看看上面例子中的数据:

| score | 0.9  | 0.8  | 0.7  | 0.6  | 0.3  | 0.2  | 0.1  |
| :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| label |  0   |  1   |  1   |  1   |  0   |  0   |  0   |

从上表中可以看到, 最大score=0.9的标签为0, 这里对应图中`precision=0`的情况, 这里就知道了0是怎么来的: 数据中有存在最高概率为Negative的数据.

**这里可以做个扩展, 理想情况下, `PR`曲线从右向左, `precision`应该是越来越大的, 如果出现了减小或者变为0的情况, 可看看对应阈值下的数据是否存在标签有误, 或者是困难样本**.

### 解决方法

最好的方法, 是通过PR曲线分析是否存在标签有错误的样本或者困难样本, 然后对测试样本进行调整.

这里有2个折中的解决方法, 可以去除这种突变:

一是限制显示范围:

```python
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
```

一是把最后一个数据去除:

```python
precision = precision[:-1]
recall = recall[:-1]
```

## 问题的延伸

对于上述问题, 是由于**负样本(示例中的`0.9(0)`)有很高的概率**, 那么**正样本有很低的概率**会是什么情况呢? 刚好有人问我这个问题, 下面以下表的例子进行测试看看.

| score | 0.9  | 0.8  | 0.7  | 0.6  | 0.3  | 0.2  | 0.1  |
| :---: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| label |  1   |  1   |  1   |  0   |  0   |  0   |  1   |

结果如下图所示:

![pr_low_porb_positive](https://gitee.com/yfor1008/pictures/raw/master/pr_low_porb_positive.png)

可以看到, 由于正样本有很低的概率(阈值), 会导致`tps`降低, 从而导致`PR`曲线存在异常.

## PR 与 ROC(Receiver Operating Characteristics)曲线

### 相互关系

有文章已经证明, PR 和 ROC 可以相互转换:

> Theorem 3.1. For a given dataset of positive and negative examples, there exists a one-to-one correspon- dence between a curve in ROCspace and a curve in PR space, such that the curves contain exactly the same confusion matrices, if Recall != 0

详见: [The Relationship Between Precision-Recall and ROC Curves](https://dl.acm.org/doi/10.1145/1143844.1143874), 网上也有很多资料有详细的说明, 下图为二者的变化趋势:

![](https://gitee.com/yfor1008/pictures/raw/master/pr_roc.gif)

### 优劣

PR 和 ROC 的区别主要在于不平衡数据的表现: PR对数据不平衡是敏感的, 正负样本比例变化会引起PR发生很大的变化; 而ROC曲线是不敏感的, 正负样本比例变化时ROC曲线变化很小. 如下图所示为不同比例正负样本情况下PR和ROC的变化:

![](https://gitee.com/yfor1008/pictures/raw/master/pr_roc_samples.gif)

ROC曲线变化很小的原因分析: tpr=TP/P, fpr=FP/N, 可以看到其计算都是在类别内部进行计算的, 只要数据内部的比例不发生变化, ROC也不会发生变化.



参考:

1. [分类模型评估之ROC-AUC曲线和PRC曲线_皮皮blog-CSDN博客_auc曲线](https://blog.csdn.net/pipisorry/article/details/51788927)

2. [精确率、召回率、F1 值、ROC、AUC 各自的优缺点是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/30643044/answer/224360465)
