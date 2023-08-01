#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

denis = cv2.imread('denis_mukwege.jpg' ,0) #copy the path of image
solvay= cv2.imread('Solvay_conference_1927.jpg',0) #copy the path of image
#plt.imshow(nadia)

plt.imshow(solvay , cmap='gray')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#detecting face 
def detect_face(img):
    faceimg= img.copy()
    face_rects = face_cascade.detectMultiScale( faceimg )
    for (x,y,w,h) in face_rects:
        cv2.rectangle(faceimg ,(x,y) , (x+w , y+h),(255,255,255), 2)
        
    return faceimg
result = detect_face(solvay)
plt.imshow(result , cmap='gray')


#rectange formed around face when detected
def adj_detect_face(img):
    faceimg= img.copy()
    face_rects = face_cascade.detectMultiScale( faceimg , scaleFactor =1.2 , minNeighbors=5)
    for (x,y,w,h) in face_rects:
        cv2.rectangle(faceimg ,(x,y) , (x+w , y+h),(255,255,255), 2)
        
    return faceimg
result = adj_detect_face(solvay)
plt.imshow(result , cmap='gray')

