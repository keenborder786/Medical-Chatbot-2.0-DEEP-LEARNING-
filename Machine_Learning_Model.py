# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:32:45 2020

@author: MMOHTASHIM
"""
from data_creation import *
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse


def plot_cm(y_true, y_pred, title):
    ''''
    input y_true-Ground Truth Labels
          y_pred-Predicted Value of Model
          title-What Title to give to the confusion matrix
    
    Draws a Confusion Matrix for better understanding of how the model is working
    
    return None
    

    '''
    
    figsize=(28,28)
    y_pred = y_pred.astype(str)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.savefig('BernoulliNB_Performance.png')
def main_model(X_train,y_train,train=True):
    '''' 
    Input--X_train-Input Feature Array(Sympothms Descriptor for each diease)
           y_train-Ground Truth Label(Diease Name)
           train-Boolean Variable to wether train the new model or not
    
    Training a new Machine Learning Model and save the model
    
    return None
    
    '''
    main=DecisionTreeClassifier()
    if train:
        main.fit(X_train,y_train)

    with open('NB.pickle','wb') as file:
        pickle.dump(main,file)
        

def machine_learning_metric_testing(X_test,y_test,matrix):
    '''' 
    Input--X_test-Input Feature Array(Sympothms Descriptor for each diease)-Testing
           y_test-Ground Truth Label(Diease Name)-Testing
           matrix-Boolean Variable to whether draw a confusion matrix
    

    return None
    '''
    
    with open('NB.pickle','rb') as file:
        model=pickle.load(file)
    y_pred=model.predict(X_test)
    if matrix:
        plot_cm(y_test, y_pred, 'Confusion Matrix for BernoulliNB')
    print("The accuracy of the model is {}".format(model.score(X_test,y_test)))


if __name__=="__main__":
    
    parser=argparse.ArgumentParser(description="Main Machine Learning Model")
    parser.add_argument('-t','--MATRIX',type=bool,help="Whether you want to create a confusion matrix")
    
    args=parser.parse_args()    
    
    X,y,df=data_machine_learning(load=True)
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1)
    main_model(X_train,y_train,train=True)
    machine_learning_metric_testing(X_test,y_test,matrix=args.MATRIX)
    
    

        