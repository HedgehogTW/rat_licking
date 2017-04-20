# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import os.path as path
import numpy as np
import pandas as pd
import cv2
from time import time

from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FormatLabel, Percentage, ProgressBar, RotatingMarker, \
SimpleProgress, Timer

MIN_DIFF = 15

def frame_diff(video_filename, mid_line, showVideo = False):
    print ('opencv version ', cv2.__version__)
    cap = cv2.VideoCapture(video_filename)
    bOpenVideo = cap.isOpened()
    print('Open Video: {0} '.format(bOpenVideo))
    if bOpenVideo == False:
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    sizeL = mid_line * height
    sizeR = (width-mid_line) * height
    
    bVideoRead, frame_p5 = cap.read() 
    bVideoRead, frame_p4 = cap.read()   
    bVideoRead, frame_p3 = cap.read()  
    bVideoRead, frame_p2 = cap.read()  
    bVideoRead, frame_p1 = cap.read()  
 
    channel = frame_p1.shape[2]
    print('fps {}, w {}, h {}, channels {}, frameCount {}'.format(
            fps, width, height, channel, frame_count))
   
#    outName = 'd:/tmp/mask.avi' 
#    vidw = cv2.VideoWriter(outName, cv2.VideoWriter_fourcc(*'XVID'), fps//2, (width, height), True)  # Make a video
#    bVideoWR = vidw.isOpened()
#    print ('VideoWriter: %d, frame %d' % (bVideoWR, frame_count))
    frame_p5 = cv2.cvtColor(frame_p5, cv2.COLOR_BGR2GRAY)
    frame_p4 = cv2.cvtColor(frame_p4, cv2.COLOR_BGR2GRAY)
    frame_p3 = cv2.cvtColor(frame_p3, cv2.COLOR_BGR2GRAY)
    frame_p2 = cv2.cvtColor(frame_p2, cv2.COLOR_BGR2GRAY)
    frame_p1 = cv2.cvtColor(frame_p1, cv2.COLOR_BGR2GRAY)

    frame_p5 = frame_p5.astype(np.int16)
    frame_p4 = frame_p4.astype(np.int16)
    frame_p3 = frame_p3.astype(np.int16)
    frame_p2 = frame_p2.astype(np.int16)
    frame_p1 = frame_p1.astype(np.int16)
   
    resultL = np.full((frame_count, 10), -1, np.float32)
    resultR = np.full((frame_count, 10), -1, np.float32)

    resultL[:5, :] = 0
    resultR[:5, :] = 0
    resultL[:5, 0] = range(5)
    resultR[:5, 0] = range(5)
    
    frameNum = 5 # start from 0
    
    widgets = [Percentage(), Bar()]
    pbar = ProgressBar(widgets=widgets, maxval=frame_count).start()
    
    while True:
        bVideoRead, frame = cap.read()  
        if bVideoRead == False:
            break
         
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.int16)
        
        diff_p5 = abs(frame - frame_p5)
        diff_p4 = abs(frame - frame_p4)
        diff_p3 = abs(frame - frame_p3)
        diff_p2 = abs(frame - frame_p2)
        diff_p1 = abs(frame - frame_p1)
        
        if showVideo:
            mask_p5 = (diff_p5 > MIN_DIFF) * 255
            mask_p4 = (diff_p4 > MIN_DIFF) * 255
            mask_p3 = (diff_p3 > MIN_DIFF) * 255
            mask_p2 = (diff_p2 > MIN_DIFF) * 255
            mask_p1 = (diff_p1 > MIN_DIFF) * 255

            mask_p1 = mask_p1.astype(np.uint8)
            mask_p2 = mask_p2.astype(np.uint8)
            mask_p3 = mask_p3.astype(np.uint8)
            mask_p4 = mask_p4.astype(np.uint8)
            mask_p5 = mask_p5.astype(np.uint8)
            
            cv2.imshow('OutputP1', mask_p1)
            cv2.imshow('OutputP2', mask_p2)
            cv2.imshow('OutputP3', mask_p3)
            cv2.imshow('OutputP4', mask_p4)     
            cv2.imshow('OutputP5', mask_p5)   
            if cv2.waitKey(100) == 27:
                break
#            mask_p1 = cv2.cvtColor(mask_p1, cv2.COLOR_GRAY2BGR)
#            mask_p1 = cv2.cvtColor(mask_p1, cv2.COLOR_GRAY2BGR)
#            mask_p1 = cv2.cvtColor(mask_p1, cv2.COLOR_GRAY2BGR)
#            mask_p1 = cv2.cvtColor(mask_p1, cv2.COLOR_GRAY2BGR)
            
#        vidw.write(mask8)
        
      
        diff_p5L = diff_p5[:, :mid_line]
        diff_p5R = diff_p5[:, mid_line:-1]     
        diff_p4L = diff_p4[:, :mid_line]
        diff_p4R = diff_p4[:, mid_line:-1]
        diff_p3L = diff_p3[:, :mid_line]
        diff_p3R = diff_p3[:, mid_line:-1]
        diff_p2L = diff_p2[:, :mid_line]
        diff_p2R = diff_p2[:, mid_line:-1]
        diff_p1L = diff_p1[:, :mid_line]
        diff_p1R = diff_p1[:, mid_line:-1]        
 
        diffSum_p5L = np.sum(diff_p5L) / sizeL      
        diffSum_p4L = np.sum(diff_p4L) / sizeL
        diffSum_p3L = np.sum(diff_p3L) / sizeL           
        diffSum_p2L = np.sum(diff_p2L) / sizeL
        diffSum_p1L = np.sum(diff_p1L) / sizeL
   
        diffSum_p5R = np.sum(diff_p5R) / sizeR     
        diffSum_p4R = np.sum(diff_p4R) / sizeR
        diffSum_p3R = np.sum(diff_p3R) / sizeR           
        diffSum_p2R = np.sum(diff_p2R) / sizeR
        diffSum_p1R = np.sum(diff_p1R) / sizeR
   
        nonzero_p5L = np.count_nonzero(diff_p5L > MIN_DIFF) / sizeL     
        nonzero_p4L = np.count_nonzero(diff_p4L > MIN_DIFF) / sizeL
        nonzero_p3L = np.count_nonzero(diff_p3L > MIN_DIFF) / sizeL
        nonzero_p2L = np.count_nonzero(diff_p2L > MIN_DIFF) / sizeL
        nonzero_p1L = np.count_nonzero(diff_p1L > MIN_DIFF) / sizeL
 
        nonzero_p5R = np.count_nonzero(diff_p5R > MIN_DIFF) / sizeR
        nonzero_p4R = np.count_nonzero(diff_p4R > MIN_DIFF) / sizeR
        nonzero_p3R = np.count_nonzero(diff_p3R > MIN_DIFF) / sizeR
        nonzero_p2R = np.count_nonzero(diff_p2R > MIN_DIFF) / sizeR
        nonzero_p1R = np.count_nonzero(diff_p1R > MIN_DIFF) / sizeR
        
        resultL[frameNum, 0] = diffSum_p1L
        resultL[frameNum, 1]= diffSum_p2L
        resultL[frameNum, 2]= diffSum_p3L
        resultL[frameNum, 3]= diffSum_p4L
        resultL[frameNum, 4]= diffSum_p5L
        resultL[frameNum, 5]= nonzero_p1L
        resultL[frameNum, 6]= nonzero_p2L
        resultL[frameNum, 7]= nonzero_p3L
        resultL[frameNum, 8]= nonzero_p4L
        resultL[frameNum, 9] = nonzero_p5L
                          
        resultR[frameNum, 0] = diffSum_p1R
        resultR[frameNum, 1]= diffSum_p2R
        resultR[frameNum, 2]= diffSum_p3R
        resultR[frameNum, 3]= diffSum_p4R
        resultR[frameNum, 4]= diffSum_p5R
        resultR[frameNum, 5]= nonzero_p1R
        resultR[frameNum, 6]= nonzero_p2R
        resultR[frameNum, 7]= nonzero_p3R
        resultR[frameNum, 8]= nonzero_p4R
        resultR[frameNum, 9] = nonzero_p5R
                          
        frame_p5 = frame_p4.copy()       
        frame_p4 = frame_p3.copy()
        frame_p3 = frame_p2.copy()
        frame_p2 = frame_p1.copy()
        frame_p1 = frame.copy()
        
        frameNum += 1
        pbar.update(frameNum)
        
#        if frameNum > 20000:
#            break
        
    pbar.finish()    
    
#    mask = resultL[:, 0] >=0
    outputL = resultL[:frameNum]
    outputR = resultR[:frameNum]
    
    header = ['diffSum_p1', 'diffSum_p2', 'diffSum_p3', 
              'diffSum_p4', 'diffSum_p5', 'nonzero_p1', 'nonzero_p2', 
              'nonzero_p3', 'nonzero_p4', 'nonzero_p5']
    df_left = pd.DataFrame(outputL, columns = header )
    df_right = pd.DataFrame(outputR, columns = header )
   
    freqStr = '{0:d}L'.format(int(1000 /fps))
    video_time = pd.timedelta_range('0:0:0', periods=frameNum, freq=freqStr)
    
    ser_time = pd.Series(video_time)   
    df_left.insert(len(df_left.columns), 'time', ser_time)
    df_right.insert(len(df_right.columns), 'time', ser_time)
    
    (root_name, ext) = path.splitext(video_filename)
    (root_name, ext) = path.splitext(root_name)
    out_nameL = '{}_L.csv'.format(root_name)
    out_nameR = '{}_R.csv'.format(root_name)
    
    df_left.to_csv(out_nameL)
    df_right.to_csv(out_nameR)
    
#    out_nameR = root_name + '_R.csv'
#    np.savetxt(out_nameL, outputL, fmt = '%.4f', delimiter=',', header = headerStr)
#    np.savetxt(out_nameR, outputR, fmt = '%.4f', delimiter=',', header = headerStr)
    print('output: {}'.format(out_nameL))
    print('output: {}'.format(out_nameR))
    return (fps, width, height)
 

    
t1 = time()    
video_file = '../../image_data/ratavi_3/930219-B-car-3-1d.avi.mkv'    
frame_diff(video_file, 183, showVideo=False)
t2 = time()
print('Computation time takes %f seconds' % (t2-t1))
sys.stdout.write('\a')

