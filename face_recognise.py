import cv2

  
import numpy as np
import face_recognition
import pickle
import os
import cvzone




cap=cv2.VideoCapture(0)
cap.set(3,1080)
cap.set(4,1400)


#load the encode
file=open('encodefile.p','rb')
encodelistwithids=pickle.load(file)
file.close()
encodelistknown,name=encodelistwithids
#encoded loaded

while True:
    success , img=cap.read()
    img=cv2.flip(img,1)

    faceloc=face_recognition.face_locations(img)
    facencode=face_recognition.face_encodings(img,faceloc)

    for encodeface,Faceloction in zip(facencode,faceloc):
        match=face_recognition.compare_faces(encodelistknown,encodeface)
        facedis=face_recognition.face_distance(encodelistknown,encodeface)

        #print(match)
        #print(facedis)
        matchindex=np.argmin(facedis)
        #print(matchindex)
        name=name[matchindex]
        #print(name)

        y1,x2,y2,x1=Faceloction
        bbox=x1,y1,x2-x1,y2-y1
        img=cvzone.cornerRect(img,bbox,rt=0)
        if match[matchindex]:
            cv2.putText(img,name,(x2-130,y2),cv2.FONT_ITALIC,1.5,(5,94,255),3)
        else:
            cv2.putText(img,"Unidentified "
                            "+", (y2 - y1, x2 - x1), cv2.FONT_ITALIC, 1.5, (5, 94, 255), 3)



    cv2.imshow("Window",img)
    if cv2.waitKey(1)==ord("q") :
        break