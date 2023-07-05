from sklearn.metrics import roc_curve, auc
# 数据准备
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# 读取你要绘制的文件
a = np.loadtxt('./result/DBFormer/AUClr0.0001dp0.2/15.csv')
# a = np.loadtxt('./result2/test_submit_FCAFormerlr0.0001dp0.2_6.txt')
# a = np.loadtxt('./fcga/result/modelFCGAFormerlr0.0001dp0.2_19.txt')
# 19 98.3
# 第1列是分数
y_score = a[:, 0]  # 取预测标签为1的概率
# 第2列是标签
y_true = a[:, 1]  # 取预测标签为1的概率
# y = np.array([1, 1, 2, 2])
# scores = np.array([0.1, 0.2, 0.75, 0.8])

# roc_curve的输入为
# y_true: 样本标签
# y_score: 模型对样本属于正例的概率输出
# pos_label: 标记为正例的标签，本例中标记为1的即为正例
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
# auc的输入为很简单，就是fpr, tpr值
auc = metrics.auc(fpr, tpr)

plt.figure(dpi=100)
lw = 2
plt.plot(fpr, tpr, color=(252 / 255, 163 / 255, 17 / 255), alpha=1,
         lw=lw, label='ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color=(19 / 255, 33 / 255, 60 / 255), lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC of the DBFormer in the bi-classification dataset')
plt.legend(loc="lower right")
plt.show()
