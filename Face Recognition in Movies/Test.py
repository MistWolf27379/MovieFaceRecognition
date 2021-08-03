import cv2
from face_recognition.api import compare_faces, face_locations
import numpy as np
import face_recognition

imgRyan=face_recognition.load_image_file('Face Recognition in Movies/ImagesTest/RyanGosling1.jpg')
imgRyan=cv2.cvtColor(imgRyan,cv2.COLOR_BGR2RGB)
imgRyanTest=face_recognition.load_image_file('Face Recognition in Movies/ImagesTest/RyanGosling2.jpg')
imgRyanTest=cv2.cvtColor(imgRyanTest,cv2.COLOR_BGR2RGB)

faceLoc= face_recognition.face_locations(imgRyan)[0]
encodeRyan= face_recognition.face_encodings(imgRyan)[0]
cv2.rectangle(imgRyan,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,0),2)

faceLocTest= face_recognition.face_locations(imgRyanTest)[0]
encodeRyanTest= face_recognition.face_encodings(imgRyanTest)[0]
cv2.rectangle(imgRyanTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,0),2)

results= face_recognition,compare_faces([encodeRyan],encodeRyanTest)
faceDis=face_recognition.face_distance([encodeRyan],encodeRyanTest)
print(results,faceDis)

cv2.imshow('Ryan Gosling',imgRyan)
cv2.imshow('Ryan Gosling Test',imgRyanTest)
cv2.waitKey(0)