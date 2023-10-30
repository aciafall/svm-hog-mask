# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:39:58 2016

@author: ldy
"""

from sklearn.svm import LinearSVC
import joblib
import glob
import os
from config import *
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

def train_svm():
    pos_feat_path = '../data/features/pos'
    neg_feat_path = '../data/features/neg'

    # Classifiers supported
    clf_type = 'LIN_SVM'

    fds = []
    labels = []
    # Load the positive features
    for feat_path in glob.glob(os.path.join(pos_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(1)

    # Load the negative features
    for feat_path in glob.glob(os.path.join(neg_feat_path,"*.feat")):
        fd = joblib.load(feat_path)
        fds.append(fd)
        labels.append(0)
    print (np.array(fds).shape,len(labels))
    
    # 数据划分为训练集和测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(fds, labels, test_size=0.3, random_state=0)
    print (np.array(X_train).shape,len(y_train))
    
    # 训练SVM并计算模型准确率 recall和precision画图表示
    clf = LinearSVC(C=0.1, penalty='l2', dual=False)
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print (recall, precision)
    print (clf.score(X_test, y_test))
    joblib.dump(clf, "../data/modelsTest/svm.pkl")
    print( "Classifier saved to {}".format(model_path))
    
    # 改变C值，观察模型准确率的变化，画图表示
    # c_range = np.linspace(0.001, 1, 20)
    # recall_list = []
    # precision_list = []
    # accuracy_list = []
    # for c in c_range:
    #     clf = LinearSVC(C=c, penalty='l2', dual=False)
    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)
    #     recall = recall_score(y_test, y_pred)
    #     precision = precision_score(y_test, y_pred)
    #     accuracy = clf.score(X_test, y_test)
    #     recall_list.append(recall)
    #     precision_list.append(precision)
    #     accuracy_list.append(accuracy)
    # plt.plot(c_range, recall_list, label='recall')
    # plt.plot(c_range, precision_list, label='precision')
    # plt.plot(c_range, accuracy_list, label='accuracy')
    # plt.xlabel('C')
    # plt.ylabel('value')
    # plt.legend()
    # plt.show()
 
    
    # if clf_type is "LIN_SVM":
    #     clf = LinearSVC()
    #     print ("Training a Linear SVM Classifier")
    #     clf.fit(fds, labels)
    #     # If feature directories don't exist, create them
    #     if not os.path.isdir(os.path.split(model_path)[0]):
    #         os.makedirs(os.path.split(model_path)[0])
    #     joblib.dump(clf, "../data/modelsTest/svm.pkl")
        
        
#训练SVM并保存模型
train_svm()