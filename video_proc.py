# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import shutil
import os
import sys, getopt
from sys import platform as _platform
import os.path as path
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import cv2
from time import time
from scipy.stats import gaussian_kde
import pathlib
from datetime import datetime


from progressbar import Bar, Percentage, ProgressBar



   
frameNum = 0
mid_line = 180
MIN_DIFF = 15
MOVING_TH_MIN = 0.04
MOVING_TH_MAX = 1
fontFace = cv2.FONT_HERSHEY_SIMPLEX
BG_HISTORY = 500
fgmask = None
fgbg = None
element = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

def detect_forground(video_filename):
    cap = cv2.VideoCapture(video_filename)
    bOpenVideo = cap.isOpened()
    print('Open Video: {0} '.format(bOpenVideo))
    if bOpenVideo == False:
        return

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        bVideoRead, frame_src = cap.read()
        if bVideoRead == False:
            break

        fgmask = fgbg.apply(frame_src)
        medianP1 = cv2.medianBlur(fgmask, 3)
        fgmask = cv2.morphologyEx(medianP1, cv2.MORPH_OPEN, kernel)

        cv2.imshow('FG', fgmask)

        key = cv2.waitKey(300)
        if key == 27:
            break
        elif key == 32:
            cv2.waitKey(0)

    cap.release()

def find_rat_center(rat_mask, bandwidth = 1.5):
    global frameNum
    y_pos = np.nan
    x_pos = np.nan
    hoz_proj = np.sum(rat_mask, axis = 1)
    ver_proj = np.sum(rat_mask, axis = 0)  
    
    list_hoz_proj = list()
    list_ver_proj = list()

    for i in range(hoz_proj.size):
        if hoz_proj[i] > 0:
            list_hoz_proj += [i] * hoz_proj[i]

    
#    f = open('list_hoz_proj.txt', 'w')
#    f.writelines('frameNum %d\n' % (frameNum))
#    f.writelines("%s\n" % item for item in list_hoz_proj)
#    f.close() 

    len_data = len(list_hoz_proj)
    if len_data >5:
        nonzero = np.count_nonzero(hoz_proj)
        if nonzero ==1:
            y_pos = np.nonzero(hoz_proj)[0]
        else:
#            list_hoz_proj.append(np.random.randint(hoz_proj.size-1))
            density_hoz = gaussian_kde(list_hoz_proj, bw_method=bandwidth)
            xs = np.linspace(0, hoz_proj.size, hoz_proj.size)
            ys = density_hoz(xs)
            y_pos = np.argmax(ys)  
#    elif len_data >5:
#        y_pos = sum(list_hoz_proj) / len_data
    
         
        
    for i in range(ver_proj.size):
        if ver_proj[i] > 0:
            list_ver_proj += [i] * ver_proj[i]

    
#    f = open('list_ver_proj.txt', 'w')
#    f.writelines('frameNum %d\n' % (frameNum))
#    f.writelines("%s\n" % item for item in list_ver_proj)
#    f.close()

    len_data =  len(list_ver_proj)
    if len_data >5:
        nonzero = np.count_nonzero(ver_proj)
        if nonzero ==1:
            y_pos = np.nonzero(ver_proj)[0]
        else:       
#            list_ver_proj.append(np.random.randint(ver_proj.size - 1))
            density_hoz = gaussian_kde(list_ver_proj, bw_method=bandwidth)
            xs = np.linspace(0, ver_proj.size, ver_proj.size)
            ys = density_hoz(xs)
            x_pos = np.argmax(ys)  
#    elif len_data >1 :
#        x_pos = sum(list_ver_proj) / len_data    
    return (x_pos, y_pos)
      
def image_process(mask_bin):
#    mask = mask_bin * 255
    mask_u8 = mask_bin.astype(np.uint8)            
    median = cv2.medianBlur(mask_u8, 3)       
    dilated = cv2.dilate(median, element)
    return dilated

def frame_diff(video_filename, dir_out, mid_line, showVideo=False, bg_subtract=False, writevideo= False):
    global fgmask
    global fgbg
    global frameNum


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

    frame_p5 = frame_p5.astype(np.int32)
    frame_p4 = frame_p4.astype(np.int32)
    frame_p3 = frame_p3.astype(np.int32)
    frame_p2 = frame_p2.astype(np.int32)
    frame_p1 = frame_p1.astype(np.int32)
   
    resultL = np.full((frame_count, 12), -1, np.float32)
    resultR = np.full((frame_count, 12), -1, np.float32)

    resultL[:5, :] = 0
    resultR[:5, :] = 0

    vidw = None
    if writevideo:
        out_video_name = outavi_path.joinpath(dir_out.name+'_out.avi')
        vidw = cv2.VideoWriter(str(out_video_name), cv2.VideoWriter_fourcc(*'XVID'), 
                           fps, (width*2, height), True)  # Make a video
    if bg_subtract:
        fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    
    frameNum = 5 # start from 0
    bg_train_frame_count = 0
    
    widgets = [Percentage(), Bar()]
    pbar = ProgressBar(widgets=widgets, maxval=frame_count).start()
    
    while True:
        bVideoRead, frame_src = cap.read()  
        if bVideoRead == False:
            break
         
        frame = cv2.cvtColor(frame_src, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.int32)
        
        diff_p5 = abs(frame - frame_p5)
        diff_p4 = abs(frame - frame_p4)
        diff_p3 = abs(frame - frame_p3)
        diff_p2 = abs(frame - frame_p2)
        diff_p1 = abs(frame - frame_p1)
      
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
   
        mask_p5 = (diff_p5 > MIN_DIFF) 
        mask_p4 = (diff_p4 > MIN_DIFF) 
        mask_p3 = (diff_p3 > MIN_DIFF) 
        mask_p2 = (diff_p2 > MIN_DIFF) 
        mask_p1 = (diff_p1 > MIN_DIFF) 

        mask_p1x = image_process(mask_p1)
        mask_p2x = image_process(mask_p2)
        mask_p3x = image_process(mask_p3)
        mask_p4x = image_process(mask_p4)
        mask_p5x = image_process(mask_p5)

        mask_p5L = mask_p5x[:, :mid_line]
        mask_p5R = mask_p5x[:, mid_line:-1]     
        mask_p4L = mask_p4x[:, :mid_line]
        mask_p4R = mask_p4x[:, mid_line:-1]
        mask_p3L = mask_p3x[:, :mid_line]
        mask_p3R = mask_p3x[:, mid_line:-1]
        mask_p2L = mask_p2x[:, :mid_line]
        mask_p2R = mask_p2x[:, mid_line:-1]
        mask_p1L = mask_p1x[:, :mid_line]
        mask_p1R = mask_p1x[:, mid_line:-1]
            
        nonzero_p5L = np.count_nonzero(mask_p5L) / sizeL     
        nonzero_p4L = np.count_nonzero(mask_p4L) / sizeL
        nonzero_p3L = np.count_nonzero(mask_p3L) / sizeL
        nonzero_p2L = np.count_nonzero(mask_p2L) / sizeL
        nonzero_p1L = np.count_nonzero(mask_p1L) / sizeL
 
        nonzero_p5R = np.count_nonzero(mask_p5R) / sizeR
        nonzero_p4R = np.count_nonzero(mask_p4R) / sizeR
        nonzero_p3R = np.count_nonzero(mask_p3R) / sizeR
        nonzero_p2R = np.count_nonzero(mask_p2R) / sizeR
        nonzero_p1R = np.count_nonzero(mask_p1R) / sizeR
                             
        (lx, ly) = find_rat_center(mask_p1L)   
        (rx, ry) = find_rat_center(mask_p1R)
        
        resultL[frameNum, 0]= diffSum_p1L
        resultL[frameNum, 1]= diffSum_p2L
        resultL[frameNum, 2]= diffSum_p3L
        resultL[frameNum, 3]= diffSum_p4L
        resultL[frameNum, 4]= diffSum_p5L
        resultL[frameNum, 5]= nonzero_p1L
        resultL[frameNum, 6]= nonzero_p2L
        resultL[frameNum, 7]= nonzero_p3L
        resultL[frameNum, 8]= nonzero_p4L
        resultL[frameNum, 9]= nonzero_p5L
        resultL[frameNum, 10]= lx
        resultL[frameNum, 11]= ly
                          
        resultR[frameNum, 0]= diffSum_p1R
        resultR[frameNum, 1]= diffSum_p2R
        resultR[frameNum, 2]= diffSum_p3R
        resultR[frameNum, 3]= diffSum_p4R
        resultR[frameNum, 4]= diffSum_p5R
        resultR[frameNum, 5]= nonzero_p1R
        resultR[frameNum, 6]= nonzero_p2R
        resultR[frameNum, 7]= nonzero_p3R
        resultR[frameNum, 8]= nonzero_p4R
        resultR[frameNum, 9]= nonzero_p5R
        resultR[frameNum, 10]= rx
        resultR[frameNum, 11]= ry       
                          
        frame_p5 = frame_p4 #.copy()       
        frame_p4 = frame_p3 #.copy()
        frame_p3 = frame_p2 #.copy()
        frame_p2 = frame_p1 #.copy()
        frame_p1 = frame #.copy()

        if bg_subtract:
            if nonzero_p1L < MOVING_TH_MAX and nonzero_p1L > MOVING_TH_MIN and \
                        nonzero_p1R < MOVING_TH_MAX and nonzero_p1R > MOVING_TH_MIN:
                if bg_train_frame_count < BG_HISTORY:
                    fgmask = fgbg.apply(frame_src)
                    bg_train_frame_count += 1
                else:
                    break
        
        if showVideo or writevideo:
            mask_p1g = mask_p1x *255           
            out_color = cv2.cvtColor(mask_p1g, cv2.COLOR_GRAY2BGR)

            font_color = (255,255,255)
            if nonzero_p1L < MOVING_TH_MAX and nonzero_p1L > MOVING_TH_MIN and \
                nonzero_p1R < MOVING_TH_MAX and nonzero_p1R > MOVING_TH_MIN:
                font_color = (0, 255, 0)

            p1str = '{0}: L {1:.4f}       R {2:.4f}'.format(frameNum, nonzero_p1L, nonzero_p1R)
            cv2.putText(out_color, p1str, (0,20), fontFace, 0.5, font_color)
            if lx is not np.nan and ly is not np.nan:
                cv2.circle(frame_src, ((int)(lx), (int)(ly)), 4, (0, 0, 255), -1)
                cv2.circle(out_color, ((int)(lx), (int)(ly)), 4, (0, 0, 255), -1)
                
            if rx is not np.nan and ry is not np.nan:
                cv2.circle(frame_src, ((int)(rx) + mid_line, (int)(ry)), 4, (0, 0, 255), -1)
                cv2.circle(out_color, ((int)(rx) + mid_line, (int)(ry)), 4, (0, 0, 255), -1)
            
            if writevideo:
                out_frame = np.hstack([frame_src, out_color])
                vidw.write(out_frame)
            if showVideo:
                cv2.imshow('OutputP1', out_color)
                cv2.imshow('Src', frame_src)
                key = cv2.waitKey(10)
                if key == 27:
                    break
                elif key == 32:
                    cv2.waitKey(0)         

        frameNum += 1
        pbar.update(frameNum)
        
#        break
#        if frameNum > 200:
#            break
        
    pbar.finish()    
    cv2.destroyAllWindows()
    cap.release()
    if writevideo:
        vidw.release()   
        
#    mask = resultL[:, 0] >=0
    outputL = resultL[:frameNum]
    outputR = resultR[:frameNum]
    
    header = ['diffSum_p1', 'diffSum_p2', 'diffSum_p3', 
              'diffSum_p4', 'diffSum_p5', 'nonzero_p1', 'nonzero_p2', 
              'nonzero_p3', 'nonzero_p4', 'nonzero_p5', 'cx', 'cy']
    df_left = pd.DataFrame(outputL, columns = header )
    df_right = pd.DataFrame(outputR, columns = header )
   
    freqStr = '{0:d}L'.format(int(1000 /fps))
    video_time = pd.date_range(0, periods=frameNum, freq=freqStr)
    
    ser_time = pd.Series(video_time)   
    df_left.insert(len(df_left.columns), 'time', ser_time)
    df_right.insert(len(df_right.columns), 'time', ser_time)
    
#    (root_name, ext) = path.splitext(video_filename)
#    (root_name, ext) = path.splitext(root_name)
#    out_nameL = '{}_L.csv'.format(root_name)
#    out_nameR = '{}_R.csv'.format(root_name)

    mdate = dir_out.name.split('-')[0]
        
    out_nameL = str(dir_out.joinpath('_diff_L_'+mdate+'.csv'))
    out_nameR = str(dir_out.joinpath('_diff_R_'+mdate+'.csv'))
    
# https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    df_left.to_csv(out_nameL, date_format='%H:%M:%S.%f')
    df_right.to_csv(out_nameR, date_format='%H:%M:%S.%f')
    
#    out_nameR = root_name + '_R.csv'
#    np.savetxt(out_nameL, outputL, fmt = '%.4f', delimiter=',', header = headerStr)
#    np.savetxt(out_nameR, outputR, fmt = '%.4f', delimiter=',', header = headerStr)
    print('output: {}'.format(out_nameL))
    print('output: {}'.format(out_nameR))
    print('bg_train_frame_count: ', bg_train_frame_count)
    return (fps, width, height)

def process_folder(dpath, outpath):
    if not outpath.exists():
        outpath.mkdir()
        
    video_list = sorted(dpath.glob('*.mkv'))
    vlist = [i for i in video_list if 'car' in str(i)]
    num_files = len(vlist)
    for i in range(num_files) :
        fname = vlist[i].name
        ftitle = fname.rsplit('.', 2)[0]  
    
        print('process video (%d/%d): %s ' % (i+1, num_files, ftitle))
        dir_out = outpath.joinpath(ftitle)
    #    print(dir_out)
        if dir_out.exists():
            shutil.rmtree(str(dir_out))
        dir_out.mkdir()
    
    
        t1 = datetime.now()
     
        frame_diff(str(vlist[i]), dir_out, mid_line, showVideo=False, bg_subtract=False, 
                   writevideo= True)  # True False
        t2 = datetime.now()
        delta = t2 - t1
        print('Computation time takes {}'.format(delta))
        
        
video_path = None 
out_path = None 
outavi_path = None
if _platform == "linux" or _platform == "linux2": # linux
   video_path = '/home/cclee/image_data/'
   out_path = '/home/cclee/tmp/'
elif _platform == "darwin": # MAC OS X
   video_path = '/Users/CCLee/image_data/' 
   out_path = '/Users/CCLee/tmp/'
elif _platform == "win32": # Windows
   video_path = 'E:/image_data/'
   out_path = 'E:/tmp/'
       
ratavi = ['ratavi_1', 'ratavi_2','ratavi_3']
def main():
    print('len(sys.argv):', len(sys.argv))
    print ('opencv version ', cv2.__version__)
    outpath = pathlib.Path(out_path)
    outpath = outpath.joinpath('rat')
    dpath = pathlib.Path(video_path)


    outavi_path = outpath.joinpath('outavi')
    if not outavi_path.exists():
        outavi_path.mkdir()


    try:
        opts, args = getopt.getopt(sys.argv[1:], "123a")
    except getopt.GetoptError as err:
        # print help information and exit:
        print( str(err))
        print('main.py -123a')             
        return 2

    for o, a in opts:
        if o == "-1":
            print('process ...', ratavi[0])
            ratpath = dpath.joinpath(ratavi[0])
            process_folder(ratpath, outpath)
        elif o == '-2':
            print('process ...', ratavi[1])
            ratpath = dpath.joinpath(ratavi[1])
            process_folder(ratpath, outpath)
        elif o == '-3':
            print('process ...', ratavi[2])
            ratpath = dpath.joinpath(ratavi[2])
            process_folder(ratpath, outpath)   
        elif o == '-a':
            for i in ratavi:
                print('process ...', i)
                ratpath = dpath.joinpath(i)
                process_folder(ratpath, outpath)            
        else:
            return 0
    


if __name__ == "__main__":
    sys.exit(main())
    



