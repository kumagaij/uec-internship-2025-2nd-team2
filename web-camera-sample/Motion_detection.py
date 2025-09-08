import cv2
import numpy as np
pointX = 0
pointY = 0
widthX = 90
widthY = 50
rect_range_x = 340    #選択範囲の左上の座標と枠の大きさ
rect_range_y = 122
rect_range_w = 180
rect_range_h = 180
range = 20
list = [0,0,0,0,0]

cap = cv2.VideoCapture("C:/Users/intern/Downloads/testdata_gpu_video_768x576.avi")
fps = int(cap.get(cv2.CAP_PROP_FPS))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (W, H)

avg = None
# ウェブカメラの画像取得
while True:
    ret, img = cap.read()
    img = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
    img = cv2.GaussianBlur(img, ksize=(7, 7), sigmaX=6)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if avg is None:
        avg = gray.copy().astype("float")
    cv2.accumulateWeighted(gray,avg,0.60)
    frameDelta = cv2.absdiff(gray,cv2.convertScaleAbs(avg))      #背景の絶対値比較

    
    ret,thresh = cv2.threshold(frameDelta,10,255,cv2.THRESH_BINARY)  #二値化
    thresh[rect_range_y:rect_range_y+rect_range_h,rect_range_x:rect_range_x + rect_range_w] = 0 #二値化画像を選択範囲のところだけマスクしている。
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #輪郭抽出
    thresh_con = img.copy()
    
    cv2.rectangle(thresh_con,(rect_range_x,rect_range_y),(rect_range_x + rect_range_w,rect_range_y + rect_range_h),(0,0,255),cv2.LINE_4) #選択範囲の枠
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)                                  #輪郭の大きさが規定値以上なら枠をつけている。
        if area > 600:
            x,y,w,h = cv2.boundingRect(contour)
            if not( x+(w/2)>rect_range_x - range and x+(w/2)<rect_range_x+rect_range_w + range and y+(h/2)>rect_range_y - range and y+(h/2)<rect_range_y+rect_range_h+range) :
                cv2.rectangle(thresh_con,(x+pointX,y+pointY),(x+pointX + w, y+pointY + h), (0, 255, 0), cv2.LINE_4)
                danger = True
                #cv2.putText(thresh_con,"safe",(10,10),cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255))
            else:
                cv2.rectangle(thresh_con,(x+pointX,y+pointY),(x+pointX + w, y+pointY + h), (0, 255, 255), cv2.LINE_4)  
                cv2.putText(thresh_con,"danger",(10,10),cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255))

    cv2.rectangle(thresh_con,(rect_range_x,rect_range_y),(rect_range_x + rect_range_w,rect_range_y + rect_range_h),(0,0,255),cv2.LINE_4) #選択範囲の枠
    cv2.rectangle(thresh_con,(rect_range_x - range,rect_range_y - range),(rect_range_x + rect_range_w + range,rect_range_y + rect_range_h + range),(255,0,0),cv2.LINE_4)
       

    cv2.imshow("Frame1", thresh_con)

    k = cv2.waitKey(100) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()