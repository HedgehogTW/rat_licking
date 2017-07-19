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
from scipy import ndimage
import rat
import classifier

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
   video_path = 'e:/image_data/'
   out_path = 'e:/tmp/rat/'
   train_path = 'e:/tmp/rat//training_grooming/'
   
read_cols = (0, 6, 10, 11, 12)
ratavi = ['ratavi_1', 'ratavi_2','ratavi_3']

def move_outvideo():
    dpath = pathlib.Path(out_path)
    outavi_path = dpath.joinpath('outavi')
    if not outavi_path.exists():
        outavi_path.mkdir()

    ddir = [x for x in dpath.iterdir() if x.is_dir()]
    for dd in ddir:  
        outfile = dd.joinpath('_output.avi')
        print(outfile)
        if not outfile.exists():
            print('no _output.avi')
            continue
        dname = dd.name
        newname = outavi_path.joinpath(dd.name+'_out.avi')
        print(newname)
        os.rename(outfile, newname)

    
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

def label_training_data(sigma):
    print('gaussian_filter1d, sigma=',sigma)
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

    fname = '_grooming_loc.csv'
    fname1 = trpath.joinpath(fname)
    df_train.to_csv(str(fname1))


    ddir = [x for x in dpath.iterdir() if x.is_dir()]
    for dd in ddir:      
        print('Check folder: ', dd)
        for ddate in date_uq:
            if ddate in str(dd):
                print('\tprocess ', ddate)
                diffrr = '_diff_R_' + ddate + '.csv'
                diffll = '_diff_L_' + ddate + '.csv'
                rrname = dd.joinpath(diffrr)
                llname = dd.joinpath(diffll)
                df = pd.read_csv(rrname, delimiter=',',index_col=0)
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')
                        
                df['label']=0
                mask = (df_train['date']==ddate) & (df_train['lr']=='R')
                df_sel = df_train[mask]
                for row in df_sel.itertuples():
                    df.loc[row.start:row.end,'label'] = 1
                
                arr = df.loc[:,'diffSum_p1':'cy']
                smooth_arr = ndimage.gaussian_filter1d(arr, sigma = sigma, axis =0 )
                header = list(df.columns)[:-2]
                header_smoo = [i + '_smoo' for i in header ]
                df_smoo = pd.DataFrame(smooth_arr, columns = header_smoo)
                df_m = pd.concat([df, df_smoo], axis=1)
                fname1 = trpath.joinpath('_label_' + ddate + '_R.csv')
                df_m.to_csv(str(fname1))
                    
                df = pd.read_csv(llname, delimiter=',',index_col=0)
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')
                
                df['label']=0
                mask = (df_train['date']==ddate) & (df_train['lr']=='L')
                df_sel = df_train[mask]
                for row in df_sel.itertuples():
                    df.loc[row.start:row.end,'label'] = 1

                arr = df.loc[:,'diffSum_p1':'cy']
                smooth_arr = ndimage.gaussian_filter1d(arr, sigma = 1.5, axis =0 )
                header = list(df.columns)[:-2]
                header_smoo = [i + '_smoo' for i in header ]
                df_smoo = pd.DataFrame(smooth_arr, columns = header_smoo)
                df_m = pd.concat([df, df_smoo], axis=1)
                
                fname1 = trpath.joinpath('_label_' + ddate + '_L.csv')
                df_m.to_csv(str(fname1))
   
    
def generate_feature():   

    trpath = pathlib.Path(train_path)    
    label_list = sorted(trpath.glob('_label_*.csv'))

    mouse = classifier.Classifier()
    for f in label_list:        
        mouse.generate_feature(f)
    
def training():
    train_lst = ['930219_L','930219_R', '930220_L', '930220_R']
    test_lst = ['930219_R', '930220_L', '930220_R']
    
    trpath = pathlib.Path(train_path)    
    fea_list = sorted(trpath.glob('_feature_*.csv'))
    train_files = []
    for f in fea_list:
        for t in train_lst:
            if t in str(f):
                train_files.append(f)
                
    test_files = []
    for f in fea_list:
        for t in test_lst:
            if t in str(f):
                test_files.append(f)
    
    if train_files:
        mouse = classifier.Classifier()
#        mouse.train(train_files, test_files)
#        mouse.separate_train_svm(train_files)
        mouse.separate_train_random_forest(train_files)
    else:
        print('no training file')
 
    
def main():
    print('len(sys.argv):', len(sys.argv))
    
    move_outvideo()
#    generate_feature()

#    
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
#            print('generate video_clip ...')
#            video_clip();
#        elif o == '-2':
#            print('training data labeling...')
#            label_training_data()
#        elif o == '-3':
#            print('generate_feature...')
#            generate_feature()
#        elif o == '-4':
#            print('training...')
#            training()            
#        else:
#            return 0
        
if __name__ == "__main__":
    main()