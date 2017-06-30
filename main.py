# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:25:44 2017

@author: cclee
"""
import sys, getopt
from sys import platform as _platform
import pathlib
import shutil
import os
from datetime import datetime
import rat


video_path = None 
out_path = None 
if _platform == "linux" or _platform == "linux2": # linux
   video_path = '/home/cclee/image_data/ratavi_3/'
   out_path = '/home/cclee/tmp/rat/'
elif _platform == "darwin": # MAC OS X
   video_path = '/Users/CCLee/image_data/RatAVI_3/' 
   out_path = '/Users/CCLee/tmp/rat/'
elif _platform == "win32": # Windows
   video_path = 'E:/image_data/RatAVI_3/'
   out_path = 'E:/tmp/rat/'
   
read_cols = (0, 6, 10, 11, 12)

def main():
    print('len(sys.argv):', len(sys.argv))
    print('data path', out_path)    
    vpath = pathlib.Path(video_path)
    dpath = pathlib.Path(out_path)
    outdir_list = sorted([x for x in dpath.iterdir() if x.is_dir()])
    num_files = len(outdir_list)
    for i, outfolder in enumerate(outdir_list):
        fname = outfolder.name
        print('process video (%d/%d): %s ' % (i+1, num_files, fname))
        fpath = dpath.joinpath(str(outfolder)+'/diff_R.csv')
        if not fpath.exists():
            print('no diff data')
            continue
        
        t1 = datetime.now()
        mouse = rat.Rat(vpath, outfolder)
        mouse.process('diff_L.csv', read_cols)
        mouse.process('diff_R.csv', read_cols)
        
        t2 = datetime.now()
        delta = t2 - t1
        print('Computation time takes {}'.format(delta))
        print('==========================================================')
        
        
if __name__ == "__main__":
    sys.exit(main())    