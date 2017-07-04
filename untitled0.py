# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 16:16:18 2017

@author: cclee
"""
import sys, getopt
from sys import platform as _platform
import pathlib
#import shutil
import os
from datetime import datetime
import pandas as pd
import rat


video_path = None 
out_path = None 
if _platform == "linux" or _platform == "linux2": # linux
   video_path = '/home/cclee/image_data/'
   out_path = '/home/cclee/tmp/rat/'
   train_path = '/home/cclee/tmp/rat/training_grooming/'
elif _platform == "darwin": # MAC OS X
   video_path = '/Users/CCLee/image_data/' 
   out_path = '/Users/CCLee/tmp/rat/'
   train_path = '/Users/CCLee/tmp/rat/training_grooming/'
elif _platform == "win32": # Windows
   video_path = 'E:/image_data/'
   out_path = 'E:/tmp/rat/'
   train_path = 'E:/tmp/rat//training_grooming/'
   
read_cols = (0, 6, 10, 11, 12)
ratavi = ['ratavi_1', 'ratavi_2','ratavi_3']


dpath = pathlib.Path(out_path)
trpath = pathlib.Path(train_path)
train_list = sorted(trpath.glob('9*.csv'))
date_lst = []
lr_lst = []
start_lst = []
end_lst = []
for f in train_list:
    fname = f.name
    ftoken = fname.split('_', 2)
    fdate = ftoken[0]
    fend = int(ftoken[2][:-4])
    fLR=  ftoken[1][0]
    fstart = int(ftoken[1][1:])
#        print(fdate, fLR, fstart, fend)
    date_lst.append(fdate)
    lr_lst.append(fLR)
    start_lst.append(fstart)
    end_lst.append(fend)
  
date_uq = list(set(date_lst))

sr_date = pd.Series(date_lst)
sr_lr = pd.Series(lr_lst)
sr_start = pd.Series(start_lst)
sr_end = pd.Series(end_lst)
df_train = pd.DataFrame({'date':sr_date, 'lr':sr_lr, 'start':sr_start, 'end':sr_end},
                  columns=['date','lr','start','end'])

fname = '_train.csv'
fname1 = trpath.joinpath(fname)
df_train.to_csv(str(fname1))


ddir = [x for x in dpath.iterdir() if x.is_dir()]
for dd in ddir:         
    for ddate in date_uq:
        if ddate in str(dd):
            diffrr = '_diff_R_' + ddate + '.csv'
            diffll = '_diff_L_' + ddate + '.csv'
            rrname = dd.joinpath(diffrr)
            llname = dd.joinpath(diffll)
            df = pd.read_csv(rrname, delimiter=',',index_col=0)
            df['label']=0
            mask = (df_train['date']==ddate) & (df_train['lr']=='R')
            df_sel = df_train[mask]
            for row in df_sel.itertuples():
                df.loc[row.start:row.end,'label'] = 1
            
            fname1 = dd.joinpath('_diff_R_' + ddate + '_label.csv')
            df.to_csv(str(fname1))
                
            df = pd.read_csv(llname, delimiter=',',index_col=0)
            df['label']=0
            mask = (df_train['date']==ddate) & (df_train['lr']=='L')
            df_sel = df_train[mask]
            for row in df_sel.itertuples():
                df.loc[row.start:row.end,'label'] = 1
  
            fname1 = dd.joinpath('_diff_L_' + ddate + '_label.csv')
            df.to_csv(str(fname1))          