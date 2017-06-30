# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:15:22 2017

@author: cclee
"""

import numpy as np
import pandas as pd
import os.path
import cv2
from numpy.lib.stride_tricks import as_strided
from progressbar import ProgressBar, Bar, Percentage

class Rat:
    jump_rows = 5
    th_min_duration = 40
    thMin = 0.009
    thMax = 0.08
#    video_name = '930219-B-car-3-1d'
    left_right = 'L'
    mid_line = 180  
    frame_loc = 0
    width = None
    height = None
    fps = None
    
    
    def __init__(self, video_dir, out_dir):
        self.video_dir = video_dir
        self.out_dir = out_dir

    def windowed_view(self, arr, window, overlap):
        arr = np.asarray(arr)
        window_step = window - overlap
        new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                      window)
        new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                       arr.strides[-1:])
        return as_strided(arr, shape=new_shape, strides=new_strides)
    
    
    def read_data(self, filename, cols):   
        data_name =  self.out_dir.joinpath(filename)
        print('Process {} ...'.format(filename))
        self.df = pd.read_csv(str(data_name), index_col=0, usecols=cols)
#        data = np.genfromtxt(str(data_name), dtype=np.float32, skip_header=1, 
#                           delimiter=',', usecols=cols)
        self.df = self.df.fillna(method='ffill')
        self.df = self.df.fillna(method='bfill')
      
#        self.data = np.array(self.df)
#        return data
          
    
    def write_video(self, cap, start, end, fps_out, lr):
        sec = start /Rat.fps
        minutes = sec // 60    
        hh = int(minutes // 60)
        mm = int(minutes % 60)
        ss = sec - hh*60*60 - mm*60
        
        dur = (end-start) / Rat.fps
    #   print('start {}, end {}'.format(start, end))
        if lr=='L':
            w = Rat.mid_line
        else:
            w = Rat.width - Rat.mid_line
        
        outName = '{}/{:s}{:05d}_{:02d}{:02d}{:05.02f}-{:.2f}.avi'.format(
                str(self.out_dir), lr, start, hh, mm, ss, dur) 
        vidw = cv2.VideoWriter(outName, cv2.VideoWriter_fourcc(*'XVID'), 
                               fps_out, (w, Rat.height), True)  # Make a video
    
        if vidw.isOpened()==True:
            while Rat.frame_loc < start:
                bVideoRead, frame = cap.read()
                Rat.frame_loc += 1
            
            while Rat.frame_loc <= end:
                bVideoRead, frame = cap.read()
                Rat.frame_loc += 1
                if bVideoRead:
                    if lr=='L':
                        vidw.write(frame[:, 0:Rat.mid_line])
                    else:
                        vidw.write(frame[:, Rat.mid_line:Rat.width])
                else:
                    break
            vidw.release()    
        else:
            print('output video error: ' , outName)
                      
    
    def write_features(self, start, end, fps_out, lr):   
        sec = start /Rat.fps
        minutes = sec // 60    
        hh = int(minutes // 60)
        mm = int(minutes % 60)
        ss = sec - hh*60*60 - mm*60
        
    
#        idx = pd.Index(self.data[start:end+1, 0], dtype='int64')
        idx = pd.Index(np.arange(start, end+1))
        ser_nonzero_p1 = self.df.loc[start:end, 'nonzero_p1']
        ser_nonzero_p5 = self.df.loc[start:end, 'nonzero_p5']
        freqStr = '{0:d}L'.format(int(1000 /fps_out))
        time_clip = pd.date_range(0, periods=end-start+1, freq=freqStr)
        
        start_time = '2000-01-01 {}:{}:{}'.format(hh, mm, ss)
        freqStr = '{0:d}L'.format(int(1000 /Rat.fps))
    #    print(start_time, 'start{}, sec{}, ss{}, mm{}]'.format(start, sec, ss, mm) ) 
        time_video = pd.date_range(start_time, periods=end-start+1, freq=freqStr)
        
        ser_time_clip = pd.Series(time_clip, idx)
        ser_time_video = pd.Series(time_video, idx)
        df = pd.DataFrame({'nonzero_p1':ser_nonzero_p1, 
                           'nonzero_p5':ser_nonzero_p5,
                           'tm_clip':ser_time_clip,
                           'tm_video':ser_time_video})
            
        outcsvName = '{}/{:s}{:05d}_{:05d}.csv'.format(str(self.out_dir), 
                      lr, start, end) 
        df.to_csv(outcsvName, date_format='%H:%M:%S.%f')
        
    def process(self, filename, cols):
        Rat.frame_loc = 0
        (fname_name, ext) = os.path.splitext(filename)
        left_right = fname_name[-1]
        self.read_data(filename, cols=cols)
#        data_p1 = self.data[:, 1]
        data_p5 = self.df.loc[:, 'nonzero_p5']
        total_frames = len(data_p5)
        # each sample represents 5 frames (5 sec)
        moving_win = self.windowed_view(data_p5, 10, 5)
        win_mean = np.mean(moving_win, axis=1)
        label = (win_mean > Rat.thMin) & (win_mean < Rat.thMax)
        label[:Rat.jump_rows] = False
        print('win_mean.shape %d' % win_mean.shape)
        print('nonzero of label %d' % np.count_nonzero(label))
        
        label_win = self.windowed_view(label, 5, 4)
        sum_label = np.sum(label_win, axis=1)
        labelLick = (sum_label == 5)
#        labelLick = sum_label 
        labelLick1 = labelLick.copy()
        for i in range(labelLick.size):
            if labelLick[i] == True:
                labelLick1[i:i+6] = True

        for i in range(6, labelLick1.size):
            if labelLick1[i] == False:
                labelLick1[i-6:i] = False 
       
#        labelLick_file_name = '{}/_labelLick_{}.csv'.format(str(self.video_dir), left_right)
#        labelLick.tofile(labelLick_file_name, sep='\n')
#        
#        labelLick_file_name = '{}/_labelLick1_{}.csv'.format(str(self.video_dir), left_right)
#        labelLick1.tofile(labelLick_file_name, sep='\n')
#        label_file_name = '{}/_label_{}.csv'.format(str(self.video_dir), left_right)
#        label.tofile(label_file_name, sep='\n')
        
        print('label_win.shape ', label_win.shape)
        print('labelLick.shape ', labelLick.shape)
        print('sum of labelLick %d, size %d' % (np.sum(labelLick), labelLick.size))
        #plt.plot(label)
        #plt.show()
#        (head_path, vname) = os.path.split(str(self.video_dir))
        video_file = '{}/{}.avi.mkv'.format(str(self.video_dir), self.out_dir.name)
#        print ('Rat::video_file ', video_file)
        cap = cv2.VideoCapture(video_file)
        bOpenVideo = cap.isOpened()
        if bOpenVideo == False:
            print('Open Video failed')
            return
        
        Rat.fps = cap.get(cv2.CAP_PROP_FPS)
        Rat.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        Rat.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fps_out = Rat.fps //2
        freqStr = '{0:d}L'.format(int(1000 /fps_out))
        
        print('fps = %d, w %d, h %d, total_frames %d, freq_out %s' % 
              (Rat.fps, Rat.width, Rat.height, total_frames, freqStr))
        
        print('min duration %d, diff thresh [%.4f %.4f]' % 
              (Rat.th_min_duration, Rat.thMin, Rat.thMax))
        
        bVideoWR = False
        extract_clips = 0
        
        widgets = [Percentage()]
        pbar = ProgressBar(widgets=widgets, maxval=labelLick1.size).start()
        
        for i in range(5,labelLick1.size):
            frameCounter = i * 5
           
            if labelLick1[i] == True:
                #print(win_mean[i])
                if bVideoWR == False:
                    start_frame = frameCounter
                    end_frame = frameCounter
                    bVideoWR = True
                else:
                    if frameCounter < total_frames:
                        end_frame = frameCounter
                    else:
                        end_frame = total_frames 
        
            else:
                if bVideoWR == True:
                    bVideoWR = False
                    if end_frame - start_frame > Rat.th_min_duration:
                        self.write_features(start_frame, end_frame, fps_out, left_right)
                        self.write_video(cap, start_frame, end_frame, fps_out, left_right)
                        extract_clips += 1
                        
            pbar.update(i)
        
        pbar.finish()      
        print('extract_clips: ', extract_clips)
        