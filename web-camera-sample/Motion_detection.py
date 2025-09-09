import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import winsound
import threading
pointX = 0
pointY = 0

Magic_Num = 180

def play_alarm():
      winsound.Beep(1000,200)

alarm_active = [False]
alarm_thread = None      
danger_active = False

cap = cv2.VideoCapture("C:/Users/intern/Downloads/syouka_6.mp4")
custom_cascade = cv2.CascadeClassifier('C:/Users/intern/Downloads/cv/cascade/cascade.xml') #cascadeファイルの読み込み
            

fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (W, H)

avg = None
# ウェブカメラの画像取得
ret, frame = cap.read()
#frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
# ROI = cv2.selectROI('Select ROIs', frame, fromCenter = False, showCrosshair = False)
# x1 = ROI[0]
# y1 = ROI[1]
# x2 = ROI[2]
# y2 = ROI[3]
# rect_range_x = x1    #選択範囲の左上の座標と枠の大きさ
# rect_range_y = y1
# rect_range_w = x2
# rect_range_h = y2
range_num = 80
while True:
    ret, img = cap.read()
    #img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
    #img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=6)   #ガウスぼかし
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    custom_rect = custom_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=0, minSize=(1,1), maxSize=(120,120))

    if avg is None:
        avg = gray.copy().astype("float")
    cv2.accumulateWeighted(gray,avg,0.60)
    frameDelta = cv2.absdiff(gray,cv2.convertScaleAbs(avg))      #背景の絶対値比較

    
    ret,thresh = cv2.threshold(frameDelta,10,255,cv2.THRESH_BINARY)  #二値化
    #thresh[rect_range_y:rect_range_y+rect_range_h,rect_range_x:rect_range_x + rect_range_w] = 0 #二値化画像を選択範囲のところだけマスクしている。
   
    thresh_con = img.copy()
    
    #cv2.rectangle(thresh_con,(rect_range_x,rect_range_y),(rect_range_x + rect_range_w,rect_range_y + rect_range_h),(0,0,255),cv2.LINE_4) #選択範囲の枠
    
            
    
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #輪郭抽出
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)                                  #輪郭の大きさが規定値以上なら枠をつけている。
        if area > 1000:
            x,y,w,h = cv2.boundingRect(contour)
            thresh_con_r = 0   
            for x_r in range(w):
                if x + x_r-1 > W:
                    break
                for y_r in range(h):
                    if y + y_r-1 > H:
                        break
                    thresh_con_r = thresh_con_r + thresh_con[y + y_r,x + x_r,2]
            thresh_con_r_ave = thresh_con_r/(w*h)
            if len(custom_rect) > 0 and thresh_con_r_ave > 0.018 :
                for x_f,y_f,w_f,h_f in custom_rect:
                            
                    
                    if x_f + w_f/2 < x + w +Magic_Num and x_f + w_f/2> x - Magic_Num and y_f + h_f/2 < y + h + Magic_Num and y_f + h_f/2>y - Magic_Num:
                        cv2.rectangle(thresh_con,(x+pointX,y+pointY),(x+pointX + w, y+pointY + h), (0, 0, 255), cv2.LINE_4)

            #if not( x+(w/2)>rect_range_x - range and x+(w/2)<rect_range_x+rect_range_w + range and y+(h/2)>rect_range_y - range and y+(h/2)<rect_range_y+rect_range_h+range) :
            #    cv2.rectangle(thresh_con,(x+pointX,y+pointY),(x+pointX + w, y+pointY + h), (0, 255, 0), cv2.LINE_4)
            #    danger = True
            #    if not danger_active:
            #        play_alarm()
            #        danger_active = True
                  
                #cv2.putText(thresh_con,"safe",(10,10),cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255))
                    else:
                        cv2.rectangle(thresh_con,(x+pointX,y+pointY),(x+pointX + w, y+pointY + h), (0, 255, 0), cv2.LINE_4)  
                        cv2.putText(thresh_con,"danger",(10,10),cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255))
                        danger_active = False
        



    #cv2.rectangle(thresh_con,(rect_range_x,rect_range_y),(rect_range_x + rect_range_w,rect_range_y + rect_range_h),(0,0,255),cv2.LINE_4) #選択範囲の枠
    #cv2.rectangle(thresh_con,(rect_range_x - range,rect_range_y - range),(rect_range_x + rect_range_w + range,rect_range_y + rect_range_h + range),(255,0,0),cv2.LINE_4)
       

    cv2.imshow("Frame1", thresh_con)

    k = cv2.waitKey(100) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
#アラーム止める
alarm_active[0] = False
if alarm_thread is not None:
   alarm_thread.join()