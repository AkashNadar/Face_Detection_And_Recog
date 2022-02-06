import numpy as np
import cv2 as cv
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-c", "--cascade", type=str,
	default="haar_face.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

people = []
DIR = r'D:\Personal\Programming\Opencv\Faces'

for i in os.listdir(DIR):
    people.append(i)

haar_cascade = cv.CascadeClassifier(args["cascade"])

# Reading the file that are been trained
# features = np.load('features.npy', allow_pickle=True)
# labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# img = cv.imread(r'D:\Personal\Programming\Opencv\images\Anand.jpg')
img = cv.imread(args["image"])
 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# Detect the faces in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 5)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    cv.putText(img, str(people[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness = 2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness = 2)

cv.imshow('Detected Faces', img)

cv.waitKey(0)