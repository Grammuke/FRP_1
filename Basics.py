import cv2
import numpy as np
import face_recognition

imgJack = face_recognition.load_image_file('ImagesBasic/Jack Dorsey.png')
imgJack = cv2.cvtColor(imgJack, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasic/Jack Test.png')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgJack)[0]
encodeJack = face_recognition.face_encodings(imgJack)[0]
cv2.rectangle(imgJack, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeJack],encodeTest)
faceDis = face_recognition.face_distance([encodeJack],encodeTest)
print(results,faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Jack Dorsey',imgJack)
cv2.imshow('Jack Test',imgTest)
cv2.waitKey(0)
