import os

import cv2
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facerecognition-c8288-default-rtdb.firebaseio.com/",
    'storageBucket': "facerecognition-c8288.appspot.com"
})

# Importing the tests images.
folderPath = 'Images'
nodePathList = os.listdir(folderPath)
imgList = []
studentsIds = []

# Here, path is the name of the files that is includes in the folderPath(Images)
for path in nodePathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentsIds.append(os.path.splitext(path)[0])  # split up into two parts, and we are getting the first one (Ids)

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


print(studentsIds)


# Encoding every single image
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # converting RGB to BGR
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encode Starting...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentsIds]
print(encodeListKnownWithIds)
print("Encode Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Successfully saved!")
