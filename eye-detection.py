import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

nadia = cv2.imread('Nadia.jpg',0) #copy the path

eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_eye(img):
    faceimg= img.copy()
    eye_rects = eye_cascade.detectMultiScale( faceimg , scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in eye_rects:
        cv2.rectangle(faceimg ,(x,y) , (x+w , y+h),(255,255,255), 1)
        
    return faceimg
result = detect_eye(nadia)
plt.imshow(result , cmap='gray')
