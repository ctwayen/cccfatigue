import cv2
import dlib
import numpy as np
import argparse


def cal_eye_dist(img):
    detector = dlib.get_frontal_face_detector() #跟opencv的haar作用一样，用来检测人脸
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) < 1:
        return 0
    landmarks = predictor(image=gray, box=faces[0]) #传入模型得到结果
    left_one = cal_e_dist(landmarks.part(37).x, landmarks.part(41).x,
                         landmarks.part(37).y, landmarks.part(41).y)/img.shape[0]
    left_two = cal_e_dist(landmarks.part(38).x, landmarks.part(40).x,
                         landmarks.part(38).y, landmarks.part(40).y)/img.shape[0]
    right_one = cal_e_dist(landmarks.part(43).x, landmarks.part(47).x,
                         landmarks.part(43).y, landmarks.part(47).y)/img.shape[0]
    right_two = cal_e_dist(landmarks.part(44).x, landmarks.part(46).x,
                         landmarks.part(44).y, landmarks.part(46).y)/img.shape[0]
    return round(left_one,4), round(left_two,4), round(right_one,4), round(right_two,4)

def cal_e_dist(x1, x2, y1, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)