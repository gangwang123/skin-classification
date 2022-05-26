#%%
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from config.cfg import train_path,test_path,val_path,labels
WIDTH, HEIGHT = 224,224
from vgg16_block import vgg16


def read_all_pic(filename):
    x_array, y_label = [], []
    for file in os.listdir(filename):
        for img in os.listdir(os.path.join(filename, file)):
            image = Image.open(os.path.join(os.path.join(filename, file), img))
            image=image.resize((WIDTH, HEIGHT))
            img_array = np.array(image) / 255
            x_array.append(img_array)
            yyyy = np.zeros((1, len(labels)))[0]
            yyyy[labels.index(file)] = 1
            y_label.append(yyyy)
    return x_array, y_label


Model=vgg16()
Model.build(input_shape=(None,WIDTH, HEIGHT, 3))
Model.load_weights('./logs/final.h5')

#%%
x_test,y_test=read_all_pic('F:\dataSet/testSet/')

#%%
print(x_test[0].shape)

#%%
print(Model.summary())

#%%
a=Model.predict(np.array(x_test))
print(tf.argmax(a,1))

#%%
np.argmax(y_test,1)
#%%
predict_label=np.array(tf.argmax(a,1))
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
predict_accuracy=accuracy_score(predict_label,np.argmax(y_test,1))
precision, recall, fscore, _ = precision_recall_fscore_support(predict_label,np.argmax(y_test,1), average='macro')
print("precision: ", precision)
print("recall: ", recall)
print("f1: ", fscore)
print("acc: ", predict_accuracy)

#%%
##评价指标的计算
classes=list(set(np.argmax(y_test,1)))
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()

#%%
# 获取混淆矩阵
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(predict_label,np.argmax(y_test,1))
plot_confusion_matrix(cm, './logs/Confusion_Matrix.png', title='test Confusion Matrix')


#%%
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


#%%
# 计算每一类的ROC
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(classes)):
    fpr[i], tpr[i], _ = roc_curve(np.array(y_test)[:, i], a[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#%%
# Compute micro-average ROC curve and ROC area（方法二）
fpr["micro"], tpr["micro"], _ = roc_curve(np.array(y_test).ravel(), np.array(a).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(classes)
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#%%
# Plot all ROC curves
lw=2
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(len(classes)), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.savefig('./logs/AUC.png')
plt.show()

#%%
print('ROC曲线下面积：',roc_auc['macro'])

#%%
from sklearn.metrics import multilabel_confusion_matrix
mcm = multilabel_confusion_matrix(predict_label,np.argmax(y_test,1),labels=classes)
#每一类的TP, FP等可以提取通过
tp = mcm[:, 1, 1]
tn = mcm[:, 0, 0]
fn = mcm[:, 1, 0]
fp = mcm[:, 0, 1]
sn=np.nan_to_num(tp/(tp+fn))
print('sensitivity:',sn)
sp=np.nan_to_num(tn/(tn+fp))
print('specificity:',sp)
prec=np.nan_to_num(tp/(tp+fp))
print('precision:',prec)

