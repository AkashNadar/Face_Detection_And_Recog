import cv2 as cv
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input image")
ap.add_argument("-c", "--cascade", type=str,
	default="haar_face.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

img = cv.imread(args["image"])
# cv.imshow('Original', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# haar_cascade = cv.CascadeClassifier('haar_face.xml')
haar_cascade = cv.CascadeClassifier(args["cascade"])

# detect how many coordinates min-neighbour - sensitivity of faces 
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors= 3)

print(f'Number of faces found {len(faces_rect)}')

# Get the coordinates of the faces 

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness= 2)

cv.imshow('Detected Faces', img)


cv.waitKey(0)