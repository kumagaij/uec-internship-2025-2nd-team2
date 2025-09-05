import cv2
import numpy as np
import sobelFilterFunction

path = "web-camera-sample/img/infinicam.png"
img = cv2.imread(path)
dst_img = img #dst = destination
#dst_img = img.copy() #dst = destination
#img_raw = cv2.imread('./images/lena.png')

# select ROIs function
ROI = cv2.selectROI('Select ROIs', img, fromCenter = False, showCrosshair = False)


x1 = ROI[0]
y1 = ROI[1]
x2 = ROI[2]
y2 = ROI[3]

print('ROI', ROI)

# Crop Image
img_crop = img[int(y1):int(y1+y2),int(x1):int(x1+x2)]

cv2.imshow("crop", img_crop)
cv2.waitKey()



# ROI領域を抜き出し、抜き出した画像をぼかす
# [top:bottom, left:right] 順序
#img_crop = img[roi[1]: roi[3], roi[0]: roi[2]]
#s_roi = cv2.blur(s_roi, (30, 30)) # ぼかし処理
img_crop = sobelFilterFunction.sobelFilter(img_crop)

cv2.imshow("sobel", img_crop)
cv2.waitKey()


#cv2.imshow("image", s_roi)
#cv2.waitKey()
#　出力画像の同じ箇所に埋め込み
dst_img[int(y1):int(y1+y2),int(x1):int(x1+x2)] = img_crop

cv2.imshow("dst", dst_img)
cv2.waitKey()


# 領域をわかりやすくするために入力画像に矩形描画
rect_img = img.copy()
cv2.rectangle(rect_img, (x1,y1), (x1+x2,y1+y2), (0, 255, 0), 2)
cv2.imshow("rect_img", rect_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

