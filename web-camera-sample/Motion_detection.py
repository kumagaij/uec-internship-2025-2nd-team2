import cv2
import numpy as np
pointX = 70
pointY = 80
widthX = 90
widthY = 50
frameNo = 0
P_count = 0
list = [0,0,0,0,0]

cap = cv2.VideoCapture("C:/Users/intern/Downloads/fire.mp4")
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
    frameNo = frameNo +1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if avg is None:
        avg = gray.copy().astype("float")
    cv2.accumulateWeighted(gray,avg,0.60)
    frameDelta = cv2.absdiff(gray,cv2.convertScaleAbs(avg))

    
    ret,thresh = cv2.threshold(frameDelta,10,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    thresh_con = img.copy()
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 300:
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(thresh_con,(x+pointX,y+pointY),(x+pointX + w, y+pointY + h), (0, 255, 0), cv2.LINE_4)
            list.pop(0)
            list.append(frameNo)
            
            
       

        cv2.imshow("Frame1", thresh_con)

    # 画像表

    k = cv2.waitKey(1) #待機時間、ミリ秒指定、0の場合はボタンが押されるまで待機
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()