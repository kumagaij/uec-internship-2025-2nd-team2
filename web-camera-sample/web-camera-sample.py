import cv2 # need to import extra module "pip install opencv-python"
import numpy as np 
from matplotlib import pyplot as plt
template = cv2.imread('ここにテンプレートの画像のパスを入れる',0)
template = cv2.cvtColor(np.array(template, dtype=np.uint8), cv2.COLOR_RGBA2BGR)
img1 = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
img2 = cv2.GaussianBlur(img1, (3, 3), 0)
sobel_x = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)               #
sobel_y = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
sobel_combined = cv2.dilate(sobel_combined, kernel)
contours, _ = cv2.findContours(sobel_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ret2, template =cv2.threshold(sobel_combined,30,255,cv2.THRESH_BINARY_INV)

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()

    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img2 = cv2.GaussianBlur(img1, (3, 3), 0)
    sobel_x = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)               #
    sobel_y = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sobel_combined = cv2.dilate(sobel_combined, kernel)
    contours, _ = cv2.findContours(sobel_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ret2, img2ti =cv2.threshold(sobel_combined,30,255,cv2.THRESH_BINARY_INV)

    w,h  = template.shape[::-1]
    res = cv2.matchTemplate(img2ti,template,cv2.TM_CCOEFF)
    min_val ,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0]+w,top_left[1]+h)
    
    cv2.rectangle(img,top_left,bottom_right,255,2)
    cv2.imshow('win1',img)
    k = cv2.waitKey(1)
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()