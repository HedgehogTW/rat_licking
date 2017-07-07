# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:18:46 2017

@author: cclee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from scipy.stats import randint as sp_randint
from datetime import datetime
from time import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

#import os.path
#import cv2
from numpy.lib.stride_tricks import as_strided
from progressbar import ProgressBar, Bar, Percentage

class Classifier:
#    def __init__(self):
#        self.video_path = video_path
#        self.tr_path = tr_path
#        self.dest_path = dest_path

    def windowed_view(self, arr, window, overlap):
        arr = np.asarray(arr)
        window_step = window - overlap
        new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                      window)
        new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                       arr.strides[-1:])
        return as_strided(arr, shape=new_shape, strides=new_strides)

    def generate_feature(self, filename):   
#        train_fname =  self.tr_path.joinpath(filename)
        if not filename.exists():
            print('no training data: ', filename)
            return
        fname = filename.name
        ftoken = fname.split('_')
        date = ftoken[2]
        lr = ftoken[3][0]
                    
        print('read training data: {} ...'.format(filename))
        df = pd.read_csv(str(filename))
        df.rename(columns = {'Unnamed: 0':'frame'}, inplace = True)
        
#        print(df.head())
        fm = df.loc[:,'frame']
        p1 = df.loc[:,'diffSum_p1']
        p5 = df.loc[:,'diffSum_p5']
        p1n = df.loc[:,'nonzero_p1']
        p5n = df.loc[:,'nonzero_p5']        
        cx = df.loc[:,'cx']
        cy = df.loc[:,'cy']
        label = df.loc[:,'label']
        
        fm_win = self.windowed_view(fm, 10, 5)
        p1_win = self.windowed_view(p1, 10, 5)
        p5_win = self.windowed_view(p5, 10, 5)
        p1n_win = self.windowed_view(p1n, 10, 5)
        p5n_win = self.windowed_view(p5n, 10, 5)        
        cx_win = self.windowed_view(cx, 10, 5)
        cy_win = self.windowed_view(cy, 10, 5)        
        label_win = self.windowed_view(label, 10, 5)
        
        fm_win_start = np.min(fm_win, axis=1)
        p1_win_mean = np.mean(p1_win, axis=1)
        p5_win_mean = np.mean(p5_win, axis=1)
        p1_win_var = np.var(p1_win, axis=1)
        p5_win_var = np.var(p5_win, axis=1)  

        p1n_win_mean = np.mean(p1n_win, axis=1)
        p5n_win_mean = np.mean(p5n_win, axis=1)
        p1n_win_var = np.var(p1n_win, axis=1)
        p5n_win_var = np.var(p5n_win, axis=1) 
        
        cx_win_mean = np.mean(cx_win, axis=1)
        cy_win_mean = np.mean(cy_win, axis=1)
        cx_win_var = np.var(cx_win, axis=1)
        cy_win_var = np.var(cy_win, axis=1)
        
        label_win_mean = np.mean(label_win, axis=1)
        label_win_mean[label_win_mean >= 0.5] = 1
        label_win_mean[label_win_mean < 0.5] = 0
        
        df1 = pd.DataFrame({'fm_start': fm_win_start, 
                            'p1_mean': p1_win_mean,
                            'p5_mean': p5_win_mean,
                            'p1_var': p1_win_var,
                            'p5_var': p5_win_var, 
                            'p1n_mean': p1n_win_mean,
                            'p5n_mean': p5n_win_mean,
                            'p1n_var': p1n_win_var,
                            'p5n_var': p5n_win_var,  
                            'cx_mean': cx_win_mean,
                            'cy_mean': cy_win_mean,                            
                            'cx_var': cx_win_var,
                            'cy_var': cy_win_var,
                            'label': label_win_mean},
            columns=['fm_start','p1_mean','p5_mean','p1_var','p5_var',\
                     'p1n_mean','p5n_mean','p1n_var','p5n_var',\
                     'cx_mean','cy_mean','cx_var','cy_var','label'])

        fname = '_feature_{}_{}.csv'.format(date, lr)
        fname1 = filename.parents[0].joinpath(fname)
        df1.to_csv(str(fname1))
        print('    generate feature file:', fname)
#        return df1
     
        
    def gaussiannb(self, Xtrain, ytrain, Xtest=None, ytest= None):
        print('process GaussianNB ...')
        print('    training data: {}, test data {}'.format(len(ytrain), len(ytest)))
#        Xtrain, Xtest, ytrain, ytest = train_test_split(xdata, ydata, random_state=1)
        self.model = GaussianNB()    
        self.model.fit(Xtrain, ytrain)  
        if Xtest:
            y_model = self.model.predict(Xtest)  
            acc = accuracy_score(ytest, y_model)
            print('    GaussianNB:', acc)

    def svm(self, Xtrain, ytrain, Xtest=None, ytest= None):
        print('process svm ...')
        print('    training data: {}'.format(len(ytrain)))
       
        t1 = datetime.now()
        
        self.model = SVC(kernel='rbf', C=1E10)
        self.model.fit(Xtrain, ytrain)  
#        cv = StratifiedShuffleSplit(n_splits=4, test_size=0.4, random_state=0)
#        scores = cross_val_score(self.model,Xtrain, ytrain, cv=cv, n_jobs=4)
#        print("CV Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        t2 = datetime.now()
        delta = t2 - t1
        print('    Computation time takes {}'.format(delta))
        y_model = self.model.predict(Xtrain)  
        acc = accuracy_score(ytrain, y_model)
        pre = precision_score(ytrain, y_model, average='binary')  
        rec = recall_score(ytrain, y_model, average='binary') 
        print('    svm rbf: train acc {:.3f}, precision {:.3f}, recall {:.3f}'.format(acc, pre, rec))
        mat = confusion_matrix(ytrain, y_model)
        print(mat)         
        if Xtest:
            y_model = self.model.predict(Xtest)  
            acc = accuracy_score(ytest, y_model)
            print('    svm rbf:', acc)

    def plot_learning_curve(self, estimator, title, X, y, train_sizes, ylim=None, cv=None,
                        n_jobs=1):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

       
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
  
        print('train_sizes', train_sizes)  
        print('train_scores_mean')
        print(train_scores_mean)
        print('test_scores_mean')
        print(test_scores_mean)         
        plt.grid()
    
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    
        plt.legend(loc="best")
        return plt
    
    def learn_curve(self, Xtrain, ytrain):
        print('process learn_curve...')
        t1 = datetime.now()
        
        title = "Learning Curves (SVM, RBF kernel)"
        estimator = SVC(kernel='rbf', C=1E10)
        cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        plt = self.plot_learning_curve(estimator, title, Xtrain, ytrain, ylim=(0.7, 1.01), 
                            train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.78, 1. ], 
                            cv=cv, n_jobs=4)
        plt.show()

        t2 = datetime.now()
        delta = t2 - t1
        print('    Computation time takes {}'.format(delta))
        
    def train(self, train_files, test_files):
        print('read train: ', train_files[0].name, end='  ')
        df_train = pd.read_csv(str(train_files[0]), delimiter=',',index_col=0)
        print(len(df_train))
        for i, tr in enumerate(train_files): 
            if i==0: continue
            print('read train:', tr.name, end='  ')
            df = pd.read_csv(str(tr), delimiter=',',index_col=0)
            print(len(df))
            df_train = df_train.append(df)
            
#        print(df_train.head(4))
        print(len(df_train), ', # of label 1:',df_train['label'].sum(axis = 0))
              
        fea_lst = ['p1_mean','p5_mean','p1_var','p5_var',\
                   'p1n_mean','p5n_mean','p1n_var','p5n_var',\
                   'cx_var','cy_var']
        print('feature:')
        print(fea_lst)
        train_sz = 0.5
        if test_files: # has test files
            Xtrain = np.asarray(df_train.loc[:,fea_lst])
            ytrain = np.asarray(df_train.loc[:,'label'])  
#            X1train, Xtest, y1train, ytest = train_test_split(Xtrain, ytrain, \
#                                                              train_size = tr_sz, random_state=1)
#            print('train_test_split, # of training ', len(X1train), tr_sz)
            cv = StratifiedShuffleSplit(n_splits=2, train_size=train_sz, random_state=0)
            train_idx, test_idx = next(iter(cv.split(Xtrain, ytrain)))
            X1train = Xtrain[train_idx]
            y1train = ytrain[train_idx]
            print('StratifiedShuffleSplit, # of training ', len(X1train))
            self.svm(X1train, y1train)
            for i, testf in enumerate(test_files): 
                print('read test: ', testf.name, end='  ')
                df_test = pd.read_csv(str(testf), delimiter=',',index_col=0)

                print(len(df_test), ', # of label 1:',df_test['label'].sum(axis = 0))
                   
                Xtest = np.asarray(df_test.loc[:,fea_lst])
                ytest = np.asarray(df_test.loc[:,'label'])              
          
                y_model = self.model.predict(Xtest)  
                acc = accuracy_score(ytest, y_model)
                pre = precision_score(ytest, y_model, average='binary')  
                rec = recall_score(ytest, y_model, average='binary') 
                print('    svm rbf: acc {:.3f}, precision {:.3f}, recall {:.3f}'.format(acc, pre, rec))
                
                mat = confusion_matrix(ytest, y_model)
                print(mat)

#                sns.heatmap(mat, square=True, annot=True, cbar=False, fmt="d")
#                plt.xlabel('predicted value')
#                plt.ylabel('true value');
#                plt.show()
        else: # no test files
            x_train = np.asarray(df_train.loc[:,fea_lst])
            y_train = np.asarray(df_train.loc[:,'label'])    
            
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
            x_idx, y_idx = next(iter(cv.split(x_train, y_train)))
            Xtrain = x_train[x_idx]
            ytrain = y_train[y_idx]
#            Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train, random_state=1)
#            self.learn_curve(x_train, y_train)
            self.svm(Xtrain, ytrain, Xtest, ytest)
               

        
#               self.svm(Xtrain, Xtest, ytrain, ytest)

    def separate_train_svm(self, train_files):
        fea_lst = ['p1_mean','p5_mean','p1_var','p5_var',\
                   'p1n_mean','p5n_mean','p1n_var','p5n_var',\
                   'cx_var','cy_var']
        print('feature:')
        print(fea_lst)
        test_sz = 0.3
        nfolds = 4
        for i, tr in enumerate(train_files): 
            print('read train: ', train_files[i].name, end='  ')
            df_train = pd.read_csv(str(train_files[i]), delimiter=',',index_col=0)
            print(len(df_train), ', # of label 1:',df_train['label'].sum(axis = 0))

            x_train = np.asarray(df_train.loc[:,fea_lst])
            y_train = np.asarray(df_train.loc[:,'label'])  
            model = SVC(kernel='rbf', C=1E10)
            
#            score = []
#            cv = StratifiedShuffleSplit(n_splits=nfolds, test_size=test_sz, random_state=0)
#            acc = cross_val_score(model, x_train, y_train, cv=cv, n_jobs=4)
#            score.append(acc.mean())
#            pre = cross_val_score(model, x_train, y_train, cv=cv, n_jobs=4, scoring='precision')
#            score.append(pre.mean())
#            rec = cross_val_score(model, x_train, y_train, cv=cv, n_jobs=4, scoring='recall')
#            score.append(rec.mean())
#            print('    CV mean accuracy {:.3f}, precision {:.3f}, recall: {:.3f}'.format(
#                    score[0], score[1], score[2]))

    
#            Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train, random_state=0)
#            print('train_test_split, # of training ', len(Xtrain))

            cv = StratifiedShuffleSplit(n_splits=nfolds, test_size=test_sz, random_state=0)
            train_idx, test_idx = next(iter(cv.split(x_train, y_train)))
            Xtrain = x_train[train_idx]
            ytrain = y_train[train_idx]
            Xtest = x_train[test_idx]
            ytest = y_train[test_idx]
            
            print('    StratifiedShuffleSplit, # of training ', len(Xtrain))
            print('    train P {}/{}, test P {}/{}'.format(np.sum(ytrain),len(Xtrain),
                  np.sum(ytest),len(Xtest)))

            model.fit(Xtrain, ytrain)  
            y_model = model.predict(Xtest)  
            acc = accuracy_score(ytest, y_model)
            pre = precision_score(ytest, y_model, average='binary')  
            rec = recall_score(ytest, y_model, average='binary') 
            print('    svm rbf: train acc {:.3f}, precision {:.3f}, recall(Sensitivity) {:.3f}'.format(acc, pre, rec))
            mat = confusion_matrix(ytest, y_model)
            specificity = mat[0,0]/(mat[0,1]+mat[0,0])
            print('    specificity: {:.3f}'.format(specificity))
            print(mat)                 
 
    # Utility function to report best scores
    def report(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")
        
    def separate_train_random_forest(self, train_files):
        # specify parameters and distributions to sample from
        param_dist = {"max_depth": [3, None],
                      "max_features": sp_randint(1, 11),
                      "min_samples_split": sp_randint(1, 11),
                      "min_samples_leaf": sp_randint(1, 11),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}

        # use a full grid over all parameters
        param_grid = {"max_depth": [3, None],
                      "max_features": [1, 3, 10],
                      "min_samples_split": [1, 3, 10],
                      "min_samples_leaf": [1, 3, 10],
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]}




        fea_lst = ['p1_mean','p5_mean','p1_var','p5_var',\
                   'p1n_mean','p5n_mean','p1n_var','p5n_var',\
                   'cx_var','cy_var']
        print('feature:')
        print(fea_lst)

        for i, tr in enumerate(train_files): 
            print('read train: ', train_files[i].name, end='  ')
            df_train = pd.read_csv(str(train_files[i]), delimiter=',',index_col=0)
            print(len(df_train), ', # of label 1:',df_train['label'].sum(axis = 0))

            X = np.asarray(df_train.loc[:,fea_lst])
            y = np.asarray(df_train.loc[:,'label']) 
            
            clf = RandomForestClassifier(n_estimators=20)
            # run randomized search
            n_iter_search = 20
            random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                               n_iter=n_iter_search)
        
        
            start = time()
            random_search.fit(X, y)
            print("RandomizedSearchCV took %.2f seconds for %d candidates"
                  " parameter settings." % ((time() - start), n_iter_search))
            self.report(random_search.cv_results_)
            
            # run grid search
            grid_search = GridSearchCV(clf, param_grid=param_grid)
            start = time()
            grid_search.fit(X, y)
            
            print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
                  % (time() - start, len(grid_search.cv_results_['params'])))
            self.report(grid_search.cv_results_)

