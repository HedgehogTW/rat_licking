# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:25:44 2017

@author: cclee
"""

import rat
import pathlib
import os.path
from datetime import datetime

read_cols = (0, 6, 10, 11, 12)

data_path = '../../image_data/ratavi_3/'
dpath = pathlib.Path(data_path)
video_list = list(dpath.glob('*.mkv'))
num_files = len(video_list)
for i in range(num_files) :
    video = video_list[i]
    file_name_noext = os.path.basename(video)
    index_of_dot = file_name_noext.index('.')
    video_name = file_name_noext[:index_of_dot]
    print('process video (%d/%d): %s ' % (i+1, num_files, video_name))
    dir_name = dpath.joinpath(video_name)
#    print(dir_out)
    if not dir_name.exists():
        print('No data folder')
        continue
    else:      
        t1 = datetime.now()
        mouse = rat.Rat(dir_name)
        mouse.process('diff_L.csv', read_cols)
        mouse.process('diff_R.csv', read_cols)
        
        t2 = datetime.now()
        delta = t2 - t1
        print('Computation time takes {}'.format(delta))
        print('==========================================================')
        
    