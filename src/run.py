import cv2
import dlib
import numpy as np
import argparse
from dlib_model import cal_eye_dist
import time

def main():
    parser = argparse.ArgumentParser(description='Running model')
    parser.add_argument('--model', type=str, default='dlib', choices=['dlib'],
                        help='model to use (default:dlib)')
    parser.add_argument('--thres', type=int, default=0.7,
                        help='threshold for fatigue (default:0.7), only when model is dlib')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for prediction (default:10), only when model is dlib')
    parser.add_argument('--cam_idx', type=int, default=0,
                        help='the index of the cam used (default:0), only when model is dlib')
    parser.add_argument('--set_time', type=int, default=3,
                        help='baseline set up time (s) (default:3), only when model is dlib')
    args = parser.parse_args()
    frame_rate = args.fps
    prev = 0
    cap = cv2.VideoCapture(args.cam_idx)
    count = 0
    baseline = []
    print(f"Start detect fatigue using dlib with camera {args.cam_idx} and fps {args.fps}. Baseline setup time is {args.set_time}s, please do not blink during the first {args.set_time}s")
    print("Notice that model will not do anything if fatigue is not detected. It will print 1 when potential fatigue is detected")
    print("press ctrl+c to exist")
    print('-------------------------------------------------------------------------------------------------------------')
    while(True):
        ret, frame = cap.read()
        time_elapsed = time.time() - prev
        #print(time_elapsed)
        if count <= args.set_time: #3s前用来设置baseline
            if time_elapsed > 1./frame_rate: #时间差大于我们设置好的1/fps时执行操作
                print(count)
                count += 1
                dst = cal_eye_dist(frame)
                baseline.append(np.mean(dst))
                prev = time.time()
            continue
        thre = np.mean(baseline) * args.thres        # using 0.6 mean as threshold. 
        if time_elapsed > 1./frame_rate: #时间差大于我们设置好的1/fps时执行操作
            dst = cal_eye_dist(frame)
            #print(dst, thre)
            if np.mean(dst) < thre:
                print(1)
            prev = time.time()
        if 0xFF == ord('q'): 
            break
    cap.release() #清楚
    
if __name__ == '__main__':
    main()
