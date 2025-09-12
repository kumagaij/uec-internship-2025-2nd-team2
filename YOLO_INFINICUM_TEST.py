import cv2 # need to import extra module "pip install opencv-python"
import pypuclib
import numpy as np
from pypuclib import CameraFactory, Camera, XferData, Decoder
from pypuclib import Resolution, PUCException, GPUSetup

from ultralytics import YOLO
import cvzone
import math
import matplotlib.pyplot as plt

#model = YOLO('yolov8n.pt')
dangerdistance = 100
#model = YOLO("yolo11n.pt")
ov_model = YOLO("yolo11n_openvino_model/")
#results = model.train(
#    data="coco8-grayscale.yaml"
#)

print(pypuclib.__doc__)

# To connect the camera first detected
cam = CameraFactory().create()

# To decode image, get decoder obj from camera
decoder = cam.decoder()

# If a GPU device is available, decoding is done on the GPU.
# To setup GPU device
reso = cam.resolution()
GPUStatus = decoder.getAvailableGPUProcess()

if GPUStatus == True:
    param = GPUSetup(reso.width, reso.height)
    decoder.setupGPUDecode(param)
    print("Decode using a GPU device")
elif GPUStatus == False:
    print("Since GPU is not available, decode using CPU")

# Set filepath to save image
savePath = "hello_world.bmp"

# Function : Save single image as BMP 
def saveBMP(img):
    cv2.imwrite(savePath, img)
    print("saved a BMP image")

# Explanation
print("press Esc to quit this application ")
print("press 's' to save a BMP image")

xferData = cam.grab()
if GPUStatus == True:
    array = decoder.decodeGPU(xferData, True, reso.width)
elif GPUStatus == False:
    array = decoder.decode(xferData)
    
# ROI = cv2.selectROI('Select ROIs', array, fromCenter = False, showCrosshair = False)
# x1 = ROI[0]
# y1 = ROI[1]
# x2 = ROI[2]
# y2 = ROI[3]
# x1 = 0
# y1 = 0 
# x2 = 0
# y2 = 0

# leftROI = x1
# topROI = y1
# rightROI = x2
# bottomROI = y2

while True:
    # Grab the single image data
    xferData = cam.grab()

    # Decode the data can be used as image
    if GPUStatus == True:
        array = decoder.decodeGPU(xferData, True, reso.width)
    elif GPUStatus == False:
        array = decoder.decode(xferData)

    # Show the image

    frame = cv2.cvtColor(array,cv2.COLOR_GRAY2BGR)
    annotated_frame = frame
    
    # Train the model on COCO8-Grayscale
    # 41: 'cup',
    results = ov_model.predict(frame)
    # print(results[0].names)
    # results = model.predict(frame, classes=[41]) # 41: 'cup' 39: 'bottle'
    
    # results = model.predict(frame, conf=0.8, classes=[0])
    
    
    annotated_frame = results[0].plot()
    # if(rightROI > 0):
    #     # cv2.rectangle(annotated_frame,(leftROI - dangerdistance,topROI - dangerdistance),(rightROI + dangerdistance,bottomROI + dangerdistance),(0,0,255),cv2.LINE_4)
    #     cv2.rectangle(annotated_frame,(x1 - dangerdistance,y1 - dangerdistance),(x1 + x2 + dangerdistance,y1 + y2 + dangerdistance),(255,0,0),cv2.LINE_4)

        
    # for box, cls in zip(results[0].boxes, results[0].boxes.cls):
    #     leftperson, topperson, rightperson, bottomperson = [int(i) for i in box.xyxy[0]]
    #     # cv2.rectangle(annotated_frame,(leftperson - dangerdistance,topperson - dangerdistance),(rightperson + dangerdistance,bottomperson + dangerdistance),(0,0,255),cv2.LINE_4)
        

    #     if(rightROI > 0):
    #         if(rightROI < leftperson):
    #             dx = leftperson - rightROI
    #         elif rightperson < leftROI:
    #             dx = leftROI - rightperson
    #         else:
    #             dx = 0
                
    #         if(bottomROI < topperson):
    #             dy = topperson - bottomROI
    #         elif bottomperson < topROI:
    #             dy = topROI - bottomperson
    #         else:
    #             dy = 0
                
    #         distance = math.sqrt(dx**2 + dy**2)
            
    #         # print(distance)
    #         cv2.putText(annotated_frame,"distance = "+str(distance),(0, 200),cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,0),thickness = 2)
            
    #         if distance < dangerdistance:
    #             cv2.putText(annotated_frame,"danger",(0, 50),cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255),thickness = 4)
    #             #cv2.imshow('Frame', annotated_frame2)
                #cv2.waitKey()
        # cv2.putText(annotated_frame2,str(distance),(0, 200),cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,0),thickness = 2)
        # cv2.imshow('Frame', annotated_frame2)
    
        #cv2.imshow('Frame', annotated_frame)
    
    cv2.imshow("INFINICAM", annotated_frame)
    
    
    # img_3dim = np.stack((array,)*3, -1)
    # ret2, img_thresholding = cv2.threshold(array, 0, 255, cv2.THRESH_OTSU) #Otsu法（大津の二値化）
    # contours, hierarchy = cv2.findContours(img_thresholding, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 
    # img_contour = cv2.drawContours(img_3dim, contours, -1, (0, 255, 0), 5)
    # cv2.imshow("INFINICAM", img_3dim)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('r'): # s : save image
        print("a")
        # ROI = cv2.selectROI('Select ROIs', array, fromCenter = False, showCrosshair = False)
        # x1 = ROI[0]
        # y1 = ROI[1]
        # x2 = ROI[2]
        # y2 = ROI[3]
    elif key & 0xFF == 27: # Esc : quit application
        break


# Close live image window
cv2.destroyAllWindows()

if GPUStatus == True:
    decoder.teardownGPUDecode()





