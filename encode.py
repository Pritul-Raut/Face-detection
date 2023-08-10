import cv2
import face_recognition
import os
import pickle

# importing the  photos

folderpath = 'photos'
pathlist=os.listdir(folderpath)
imglist=[]
name=[]


for path in pathlist:
    imglist.append(cv2.imread(os.path.join(folderpath,path)))
    #print(path)
    #print(os.path.splitext(path))
    name.append(os.path.splitext(path)[0])
#print(name)


def imageencode(imagelist):
    encode=[]
    for img in imagelist:

        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        imgencode=face_recognition.face_encodings(img)[0]
        encode.append(imgencode)

    return encode

#encode start
encodelistknown=imageencode(imglist)
#print(encodelistknown)
encodelistwithids=[encodelistknown,name]
#encode done

file=open("encodefile.p",'wb')
pickle.dump(encodelistwithids,file)
file.close()
print("file")





