import cv2
from face_recognition.api import compare_faces, face_locations
import numpy as np
import face_recognition
import os

path='Face Recognition in Movies/Images'
images = []
names=[]
myList= os.listdir(path)
for cls in myList:
    curImg=cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    names.append(os.path.splitext(cls)[0])

def findencodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown=findencodings(images)
vid=cv2.VideoCapture(0)

while True:
    success,video = vid.read()
    
    faceCurFrame= face_recognition.face_locations(video)
    encodeCurFrame=face_recognition.face_encodings(video,faceCurFrame)
    
    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name= names[matchIndex].upper()
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(video,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(video,(x1,y2-35),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(video,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    cv2.imshow("Video",video)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break