# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:25:44 2017

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


    
def video_clip():

    vpath = pathlib.Path(video_path)
    dpath = pathlib.Path(out_path)
    
    for v in ratavi:
        
        avipath = vpath.joinpath(v)
        video_list = sorted(avipath.glob('*.mkv'))
        vlist = [i for i in video_list if 'car' in str(i)]
        num_files = len(vlist)

        for i in range(num_files) :
            fname = vlist[i].name
            ftitle = fname.rsplit('.', 2)[0]          

            print('process {}, {}/{}, {}'.format(v, i+1, num_files, ftitle))
            
            destpath = dpath.joinpath(ftitle)
            if not destpath.exists():
                print('no destpath path')
                continue
        
            fdate = ftitle.split('-')[0]    
            diff_r = '_diff_R_'+fdate + '.csv'
            diff_l = '_diff_L_'+fdate + '.csv'
            fpath = destpath.joinpath(diff_r)
            if not fpath.exists():
                print('no diff data')
                continue
        
            files = destpath.glob('9*')
            for f in files:   os.remove(f)
            files = destpath.glob('L*')
            for f in files:   os.remove(f)
            files = destpath.glob('R*')
            for f in files:   os.remove(f)

            t1 = datetime.now()
            mouse = rat.Rat(avipath, destpath)
            mouse.process(diff_l, read_cols)
            mouse.process(diff_r, read_cols)
        
            t2 = datetime.now()
            delta = t2 - t1
            print('Computation time takes {}'.format(delta))
            print('==========================================================')

def label_training():
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
        fend = ftoken[2][:-4]
        fLR=  ftoken[1][0]
        fstart = ftoken[1][1:]
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
#    df_train.to_csv(str(fname1))


    ddir = [x for x in dpath.iterdir() if x.is_dir()]
    for dd in ddir:         
        for ddate in date_uq:
            if ddate in str(dd):
                diffrr = '_diff_R_' + ddate + '.csv'
                diffll = '_diff_L_' + ddate + '.csv'
                rrname = dd.joinpath(diffrr)
                llname = dd.joinpath(diffll)
                dfr = pd.read_csv(rrname, delimiter=',',index_col=0)
                
                print(dfr)
    
   
def main():
    print('len(sys.argv):', len(sys.argv))
    label_training()
#    try:
#        opts, args = getopt.getopt(sys.argv[1:], "1234")
#    except getopt.GetoptError as err:
#        # print help information and exit:
#        print( str(err))
#        print('main.py -1234')             
#        return 2
#
#    for o, a in opts:
#        if o == "-1":
#            print('video_clip ...')
#            video_clip();
#        elif o == '-2':
#            print('training data labeling...')
#            label_training()
#        else:
#            return 0
        
if __name__ == "__main__":
    main()