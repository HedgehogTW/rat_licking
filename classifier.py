# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:18:46 2017

@author: cclee
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 


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
        p1 = df.loc[:,'nonzero_p1']
        p5 = df.loc[:,'nonzero_p5']
        cx = df.loc[:,'cx']
        cy = df.loc[:,'cy']
        label = df.loc[:,'label']
        
        fm_win = self.windowed_view(fm, 10, 5)
        p1_win = self.windowed_view(p1, 10, 5)
        p5_win = self.windowed_view(p5, 10, 5)
        cx_win = self.windowed_view(cx, 10, 5)
        cy_win = self.windowed_view(cy, 10, 5)        
        label_win = self.windowed_view(label, 10, 5)
        
        fm_win_start = np.min(fm_win, axis=1)
        p1_win_mean = np.mean(p1_win, axis=1)
        p5_win_mean = np.mean(p5_win, axis=1)
        p1_win_var = np.var(p1_win, axis=1)
        p5_win_var = np.var(p5_win, axis=1)        
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
                            'cx_var': cx_win_var,
                            'cy_var': cy_win_var,
                            'label': label_win_mean},
            columns=['fm_start','p1_mean','p5_mean','p1_var','p5_var','cx_var','cy_var','label'])

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
        t2 = datetime.now()
        delta = t2 - t1
        print('    Computation time takes {}'.format(delta))
        
        if Xtest:
            y_model = self.model.predict(Xtest)  
            acc = accuracy_score(ytest, y_model)
            print('    svm rbf:', acc)
            
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
            
        print(df_train.head(4))
        print(len(df_train))
              
        fea_lst = ['p1_mean','p5_mean','p1_var','p5_var','cx_var','cy_var']
        
        if not test_files:
            x_train = np.asarray(df_train.loc[:,fea_lst])
            y_train = np.asarray(df_train.loc[:,'label'])        
            Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train, random_state=1)
            self.gaussiannb(Xtrain, ytrain, Xtest, ytest)
#           self.svm(Xtrain, ytrain, Xtest, ytest)
        else:                
            Xtrain = np.asarray(df_train.loc[:,fea_lst])
            ytrain = np.asarray(df_train.loc[:,'label'])  
            self.svm(Xtrain, ytrain)
            for i, testf in enumerate(test_files): 
                print('read test: ', testf.name, end='  ')
                df_test = pd.read_csv(str(testf), delimiter=',',index_col=0)
                print(len(df_test))
                   
                Xtest = np.asarray(df_test.loc[:,fea_lst])
                ytest = np.asarray(df_test.loc[:,'label'])              
          
                y_model = self.model.predict(Xtest)  
                acc = accuracy_score(ytest, y_model)
                print('    svm rbf:', acc)
        
#               self.svm(Xtrain, Xtest, ytrain, ytest)

            
             
             
             
        