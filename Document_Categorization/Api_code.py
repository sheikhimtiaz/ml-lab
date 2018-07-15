# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 17:27:25 2018

@author: User
"""

import os
import re
import operator
import sys
import csv
    
def path_discover():
    test_directory = "C:\\Users\\User\\Desktop\\ML\\Nazim\\category"
    directories = []
    for child in os.listdir(test_directory):
        test_path = os.path.join(test_directory, child)
        if os.path.isdir(test_path):
            directories.append(test_path)
        else:
            print(test_path)
        
    return directories;
            
if __name__ == "__main__":
    directories = path_discover()
    
    #print(directories)
    
    #sys.exit() 
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    
    train_size = 500
    test_size = 50
    fmap = {}
    classes = []
    for directory in directories:
        #print(directory)
        
        tlist = directory.split("\\")
#        print(tlist)
        sol_path = ""
        
        lsiz = len(tlist)
        cls = tlist[lsiz-1]
        classes.append(cls)
        
        for i in range(lsiz):
            if i+2 == lsiz:
                sol_path += "sol"
                
            else:
                sol_path += tlist[i]
                
            if i+1 != lsiz :
                sol_path += "\\"
            
        print(sol_path)
        #sol_path += "\\link"
        #directory += "\\link"
        
        if not os.path.exists(sol_path):
           os.makedirs(sol_path)
        
        gmap = {}
        
        
        for it in range(train_size+test_size):
            inp_path = directory + "\\link" + str(it+1)
            out_path = sol_path + "\\link" + str(it+1)
        
            inp_path += ".txt"
            out_path += ".txt"
#            print(inp_path)
#            print(out_path)
            
            file = open(inp_path,"rb")
            writer = open(out_path,"wb")
            p = file.read()
#            print(p)
            uni = str(p,'utf-8')
#            print(uni)
            if(it < train_size):
                x_train.append(uni)
                y_train.append(cls)
            else:
                x_test.append(uni)
                y_test.append(cls)
    
    print(classes)
    size_train = len(x_train)
    size_test = len(x_test)
    
#    for i in range(sizee):
#        doc = doc_arr[i]
#        cat = cat_arr[i]
#        print(doc)
#        print(' -----> ' , cat)
    
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.1) 
    tfidf_train = tfidf_vectorizer.fit_transform(x_train) 
    tfidf_test = tfidf_vectorizer.transform(x_test)
    
#    print(tfidf_train)
#
#    print(tfidf_vectorizer.get_feature_names()[-10:])
#
#    feature_arr = tfidf_vectorizer.get_feature_names()
#    for i in range(10):
#        val = tfidf_vectorizer.vocabulary_[feature_arr[i]]
#        print(feature_arr[i] , " ----> " , val)
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        See full source and example: 
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, with only tfidf')
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
    
    from sklearn.naive_bayes import MultinomialNB
    from sklearn import metrics
    import numpy as np
    import itertools
    
    clf_tree = MultinomialNB() 
    clf_tree.fit(tfidf_train, y_train)
    pred_tree = clf_tree.predict(tfidf_test)

    i = len(pred_tree) - 1
    cnt = 100
    score = metrics.accuracy_score(y_test, pred_tree)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred_tree, labels=classes)
    plot_confusion_matrix(cm, classes=classes)


















