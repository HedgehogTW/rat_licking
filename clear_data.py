# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:59:00 2017

@author: cclee
"""


import pathlib
import os


data_path = '../../image_data/ratavi_3/'
dpath = pathlib.Path(data_path)
sub_dir = [x for x in dpath.iterdir() if x.is_dir()]
for d in sub_dir:
    files = d.glob('L*.csv')
    for f in files:   os.remove(f)

    files = d.glob('R*.csv')
    for f in files:   os.remove(f)    
    
    files = d.glob('*.avi')
    for f in files:   os.remove(f)    
    
    files = d.glob('_*.csv')
    for f in files:   os.remove(f)    
    
    files = d.glob('label*.csv')
    for f in files:   os.remove(f)      