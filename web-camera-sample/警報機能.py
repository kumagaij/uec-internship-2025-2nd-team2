import cv2 
import numpy as np
import matplotlib.pyplot as plt
import sys
import winsound
import threading

def play_alarm():
   while alarm_active[0]:
      winsound.Beep(1000,500)
      cv2.waitKey(50)

cap = cv2.VideoCapture(0)

#繰り返しのためのwhile文
while True:
    # ウェブカメラの画像取得
    ret, img = cap.read()
    if not ret:
       break

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #グレースケール化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

       #顔を検知して音を鳴らす
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    alarm_active = [False]
    alarm_thread = None
    # # 顔を検知   
    faces = face_cascade.detectMultiScale(
      gray,
      scaleFactor=1.1,
      minNeighbors=4,
      minSize=(30,30)
    )
    #顔が映ってるとき
    if len(faces) > 0:
      cv2.putText(img, "face", (5, 430), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
      for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        if not alarm_active[0]:
           alarm_active[0] = True
           alarm_thread = threading.Thread(target=play_alarm)
           alarm_thread.start()
    else:
       if alarm_active[0]:
          alarm_active[0] = False
          if alarm_thread is not None:
             alarm_thread.join()
    # 顔認識画像表示
    cv2.imshow('Facs Detection(Color)',img)
    key=cv2.waitKey(10)
    if key==27:
       break
    
 

cap.release()
cv2.destroyAllWindows()
#アラーム止める
alarm_active[0] = False
if alarm_thread is not None:
   alarm_thread.join()