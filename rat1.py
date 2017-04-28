import sys
import pathlib
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from numpy.lib.stride_tricks import as_strided
from progressbar import AnimatedMarker, Bar, BouncingBar, Counter, ETA, \
    FormatLabel, Percentage, ProgressBar, RotatingMarker, \
SimpleProgress, Timer

jump_rows = 5
thMin = 0.004
thMax = 0.08
data_path = '../../image_data/ratavi_3/'
video_name = '930219-B-car-3-1d'
left_right = 'L'
read_cols = (0, 6, 10)
mid_line = 180  
frame_loc = 0

def windowed_view(arr, window, overlap):
    arr = np.asarray(arr)
    window_step = window - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step,
                                  window)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) +
                   arr.strides[-1:])
    return as_strided(arr, shape=new_shape, strides=new_strides)


def read_data(dir_name, filename, cols):    
    data_name =  dir_name.joinpath(filename)
    data = np.genfromtxt(str(data_name), dtype=np.float32, skip_header=1, 
                       delimiter=',', usecols=cols)
    return data


def write_video(start, end, fps_out, lr):
    global frame_loc 
    sec = start /fps
    minutes = sec // 60    
    hh = int(minutes // 60)
    mm = int(minutes % 60)
    ss = sec - hh*60*60 - mm*60
#   print('start {}, end {}'.format(start, end))
    if lr=='L':
        w = mid_line
    else:
        w = width - mid_line
    outName = '../../tmp/%c%05d_%02d%02d%05.02f.avi' % (lr, start, hh, mm, ss)
    vidw = cv2.VideoWriter(outName, cv2.VideoWriter_fourcc(*'XVID'), 
                           fps_out, (w, height), True)  # Make a video

    if vidw.isOpened()==True:
        while frame_loc < start:
            bVideoRead, frame = cap.read()
            frame_loc += 1
        
        while frame_loc <= end:
            bVideoRead, frame = cap.read()
            frame_loc += 1
            if bVideoRead:
                if lr=='L':
                    vidw.write(frame[:, 0:mid_line])
                else:
                    vidw.write(frame[:, mid_line:width])
            else:
                break
        vidw.release()    
    else:
        print('output video error: ' , outName)
                  

def write_features(start, end, fps_out, lr):   
    sec = start /fps
    minutes = sec // 60    
    hh = int(minutes // 60)
    mm = int(minutes % 60)
    ss = sec - hh*60*60 - mm*60

    idx = pd.Index(data[start:end+1, 0], dtype='int64')
    ser_nonzero_p1 = pd.Series(data[start:end+1, 1], idx)
    ser_nonzero_p5 = pd.Series(data[start:end+1, 2], idx)
    freqStr = '{0:d}L'.format(int(1000 /fps_out))
    time_clip = pd.date_range(0, periods=end-start+1, freq=freqStr)
    
    start_time = '2000-01-01 {}:{}:{}'.format(hh, mm, ss)
    freqStr = '{0:d}L'.format(int(1000 /fps))
#    print(start_time, 'start{}, sec{}, ss{}, mm{}]'.format(start, sec, ss, mm) ) 
    time_video = pd.date_range(start_time, periods=end-start+1, freq=freqStr)
    
    ser_time_clip = pd.Series(time_clip, idx)
    ser_time_video = pd.Series(time_video, idx)
    df = pd.DataFrame({'nozeroP1':ser_nonzero_p1, 
                       'nozeroP5':ser_nonzero_p5,
                       'tm_clip':ser_time_clip,
                       'tm_video':ser_time_video})
    
    outcsvName = '../../tmp/%c%05d.csv' % (lr, start)
    df.to_csv(outcsvName, date_format='%H:%M:%S.%f')
# https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior

dpath = pathlib.Path(data_path)
dir_name = dpath.joinpath(video_name)

data = read_data(dir_name, 'diff_L.csv', read_cols)
print('Load data columns: ', read_cols)
np.set_printoptions(precision=4, suppress=True)
print(data[4:7])
print('....')
print(data[-3:])

data_p1 = data[:, 1]
data_p5 = data[:, 2]
total_frames = data[-1, 0]
# each sample represents 5 frames (5 sec)
moving_win = windowed_view(data_p5, 10, 5)
win_mean = np.mean(moving_win, axis=1)
label = (win_mean > thMin) & (win_mean < thMax)
label[:jump_rows] = False
print('win_mean.shape %d' % win_mean.shape)
print('nonzero of label %d' % np.count_nonzero(label))

label_win = windowed_view(label, 5, 4)
sum_label = np.sum(label_win, axis=1)
labelLick = (sum_label == 5)
labelLick1 = labelLick.copy()
for i in range(labelLick.size):
    if labelLick[i] == True:
        labelLick1[i:i+5] = True

labelLick_file_name = '../../image_data/ratavi_3/{}_labelLick.csv'.format(video_name)
labelLick1.tofile(labelLick_file_name, sep='\n')
label_file_name = '../../image_data/ratavi_3/{}_label.csv'.format(video_name)
label.tofile(label_file_name, sep='\n')

print('label_win.shape ', label_win.shape)
print('labelLick.shape ', labelLick.shape)
print('sum of labelLick %d, size %d' % (np.sum(labelLick), labelLick.size))
#plt.plot(label)
#plt.show()

print ('opencv version ', cv2.__version__)
video_file = '../../image_data/ratavi_3/{}.avi.mkv'.format(video_name)
cap = cv2.VideoCapture(video_file)
bOpenVideo = cap.isOpened()
print('Open Video: ', bOpenVideo)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps_out = fps //2
freqStr = '{0:d}L'.format(int(1000 /fps_out))

print('fps = %d, w %d, h %d, total_frames %d, freq_out %s' % 
      (fps, width, height, total_frames, freqStr))

bVideoWR = False
vidw = cv2.VideoWriter()
extract_clips = 0

widgets = [Percentage(), Bar()]
pbar = ProgressBar(widgets=widgets, maxval=labelLick1.size).start()

for i in range(5,labelLick1.size):
    frameCounter = i * 5
   
    if labelLick1[i] == True:
        #print(win_mean[i])
        if bVideoWR == False:
            extract_clips += 1
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
            if end_frame > start_frame:
                write_features(start_frame, end_frame, fps_out, left_right)
                write_video(start_frame, end_frame, fps_out, left_right)
    
    pbar.update(i)

pbar.finish()      
print('extract_clips: ', extract_clips)
sys.stdout.write('\a')

 