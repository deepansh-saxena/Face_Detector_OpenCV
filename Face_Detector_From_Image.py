import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('D:/Users/sanch/Desktop/PythonProjects/pretrainedData.xml')

img = cv2.imread('D:/Users/sanch/Desktop/PythonProjects/bigfamily.png')


grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)  

cv2.imshow('face detector', img)
cv2.waitKey()
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img,  (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)
    print(face_coordinates)
    cv2.imshow('face detector', img)
    cv2.waitKey()


   

print('Code Completed')