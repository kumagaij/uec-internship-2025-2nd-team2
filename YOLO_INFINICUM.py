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
dangerdistance = 40
xAmin = None
xBmin = None
#model = YOLO("yolo11n.pt")

#model.export(format="openvino")  # creates 'yolo11n_openvino_model/'
# Load the exported OpenVINO model
ov_model = YOLO("yolo11n_openvino_model/")



class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


#{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


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

names = ov_model.names


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
    #results = model.predict(frame)
    results = ov_model.predict(frame)
    # print(results[0].names)
    # results = model.predict(frame, classes=[41]) # 41: 'cup' 39: 'bottle'
    
# results = model.predict(frame, conf=0.8, classes=[0])
    if results:  # 結果がある場合のみ処理を実行
        for box in results[0].boxes:
            print(box.id)
            classid = int(box.cls[0])
            confidence = float(box.conf[0])  # 信頼度
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # バウンディングボックスの座標
            
            if classid == 41: #62: 'tv'
                color = (0, 255, 0)  # 緑
                xAmin = xmin;yAmin = ymin;xAmax = xmax;yAmax = ymax
            elif classid == 0: #0: 'person'
                color = (0, 0, 255)  # 赤
                xBmin = xmin;yBmin = ymin;xBmax = xmax;yBmax = ymax                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
            else:
                color = (200, 200, 200) 

            class_name =class_names[classid]
            label = f"{classid}:{class_name}"
            cv2.rectangle(frame, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)
            if  classid == 41:
                cv2.rectangle(frame, pt1=(xmin-dangerdistance, ymin-dangerdistance), pt2=(xmax+dangerdistance, ymax+dangerdistance),color=(255,0,0),thickness=2)
            
            y = ymin - 15 if ymin - 15 > 15 else ymin + 15
            cv2.putText(frame, label, (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
            if xAmin and xBmin:
                if(xAmax < xBmin):
                    dx = xBmin - xAmax
                elif xBmax < xAmin:
                    dx = xAmin - xBmax
                else:
                    dx = 0
            
                if(yAmax < yBmin):
                    dy = yBmin - yAmax
                elif yBmax < yAmin:
                    dy = yAmin - yBmax
                else:
                    dy = 0
                    
                distance = math.sqrt(dx**2 + dy**2)
                
                # print(distance)
                cv2.putText(frame,"distance = "+str(distance),(0, 200),cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,0),thickness = 2)
                
                if distance < dangerdistance:
                    cv2.putText(frame,"danger",(0, 50),cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255))
                    
    cv2.imshow("INFINICAM", frame)
    
    

    key = cv2.waitKey(1)

    if key & 0xFF == 27: # Esc : quit application
        break


# Close live image window
cv2.destroyAllWindows()

if GPUStatus == True:
    decoder.teardownGPUDecode()





