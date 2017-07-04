# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:18:46 2017

@author: cclee
"""

import numpy as np
import pandas as pd
import os.path
import cv2
from numpy.lib.stride_tricks import as_strided
from progressbar import ProgressBar, Bar, Percentage

class Classifier:
    def __init__(self, video_path, tr_path, dest_path):
        self.video_path = video_path
        self.tr_path = tr_path
        self.dest_path = dest_path

    def windowed_view(self, arr, window, overlap):
        arr = np.asarray(arr)
        window_step = window - overlap
        new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                      window)
        new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                       arr.strides[-1:])
        return as_strided(arr, shape=new_shape, strides=new_strides)

    def generate_feature(self, filename, lr, date):   
        train_fname =  self.tr_path.joinpath(filename)
        if not train_fname.exists():
            print('no training data: ', train_fname)
            return
            
        print('read training data: {} ...'.format(filename))
        df = pd.read_csv(str(train_fname))
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
        cx_win_var = np.var(cx_win, axis=1)
        cy_win_var = np.var(cy_win, axis=1)
        
        label_win_mean = np.mean(label_win, axis=1)
        label_win_mean[label_win_mean >= 0.5] = 1
        label_win_mean[label_win_mean < 0.5] = 0
        
        df1 = pd.DataFrame({'fm_start': fm_win_start, 
                            'p1_mean': p1_win_mean,
                            'p5_mean': p5_win_mean,
                            'cx_var': cx_win_var,
                            'cy_var': cy_win_var,
                            'label': label_win_mean},
            columns=['fm_start','p1_mean','p5_mean','cx_var','cy_var','label'])

        fname = '_{}_{}_feature.csv'.format(date, lr)
        fname1 = self.tr_path.joinpath(fname)
        df1.to_csv(str(fname1))
        print('\tgenerate feature file:', fname)
        
    def train(self, train_lst):
        for tr in train_lst:
            ftoken = tr.split('_')
            lr = ftoken[2]
            date = ftoken[3]
            self.generate_feature(tr, lr, date)
             
             
             
        