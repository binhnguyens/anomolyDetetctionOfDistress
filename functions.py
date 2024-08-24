#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 21:19:19 2021

@author: binhnguyen
"""

#Loading up necessary libraries
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import numpy as np

#Loads up machine learning tools for LDA, LR, DT, and SVM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.metrics import plot_confusion_matrix

# Load files
def fileload (filename, choice):
    path = ('/Users/binhnguyen/Documents/MATLAB/2. BIOPAC/')
    # filename = ('python_FS_P1.mat')
    
    if (choice == 1):
        keys = 'python_FS'        
    elif (choice == 2):
        keys = 'label'
    elif (choice == 3):
        keys = 'mi_val'
    elif (choice == 4):
        keys = 'pss_val'

    file = (path + filename)
    feature_set = loadmat(file)



    return feature_set[keys]

# Hyper plane function
def hyperplane (clf,X,Y):
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(-1, 3)
    yy = a * xx - (clf.intercept_[0]) / w[1]
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.plot(xx, yy, "k-") #********* This is the separator line ************

    plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors="k")
    plt.show()

# Confusion matrix
def CM (y_test, y_pred,title):
    cm = confusion_matrix(y_test, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# SVM
def SVM_prediction(X_train, y_train, X_test,y_test):

    clf = svm.SVC(C=1.0, kernel='linear', max_iter=-1, decision_function_shape='ovr') # Load-up of SVM ML methodology using default setting
    clf.fit(X_train, y_train) # Create SVM model based on default settings
    y_pred = clf.predict (X_test) # SVM prediction for test values
    acc = accuracy_score(y_test, y_pred) #Accuracy of the model
    CM(y_test, y_pred,'SVM')
#     hyperplane (clf,X_test,y_test)
    
    print ('The accuracy of SVM is %f%%' %(acc*100)) #Prints accuracy

# DT
def DT_prediction(X_train, y_train, X_test,y_test):
    clf = DecisionTreeClassifier(class_weight=None, ccp_alpha=0.0) # Load-up of DT ML methodology using default setting
    clf.fit(X_train, y_train) # Create DT model based on default setting
    y_pred = clf.predict (X_test) # DT prediction for test values
    acc = accuracy_score(y_test, y_pred) #Accuracy of the model
    CM(y_test, y_pred,'DT')
    print ('The accuracy of DT is %f%%' %(acc*100)) #Prints accuracy

# LR
def LR_prediction(X_train, y_train, X_test,y_test):
    clf = LogisticRegression() # Load-up of LR ML methodology using default setting
    clf.fit(X_train, y_train) # Create LR model based on default setting
    y_pred = clf.predict (X_test) # LR prediction for test values
    acc = accuracy_score(y_test, y_pred) #Accuracy of the model
    CM(y_test, y_pred,'LR')
    print ('The accuracy of LR is %f%%' %(acc*100)) #Prints accuracy