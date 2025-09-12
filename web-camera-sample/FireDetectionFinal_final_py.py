import cv2
from ultralytics import YOLO
import threading
import tkinter as tk
import numpy as np
import math
import pygame.mixer
from ultralytics import solutions
import torch
import openvino as ov

waitkeytime = 1 #待機時間
dangerdistance = 20 #危険距離の閾値
resizescale = 0.65 #表示する画像の大きさ
App_start_Flag = True
current_index = 0

#YOLOのLOGを消す処理
from logging import getLogger
logger = getLogger('ultralytics')
logger.disabled = True

#ファイルパス
video_path = r'syouka.mp4'
mp3file_path = 'C:/Users/intern/Downloads/soundfile_test_2.mp3'
YOLOmodel_path = 'C:/Users/intern/Downloads/uec-internship-2025-2nd-team2-develop/uec-internship-2025-2nd-team2-develop/web-camera-sample/Fire_Detection-main/yolo11n.pt'
Firemodel_path = 'C:/Users/intern/Downloads/uec-internship-2025-2nd-team2-develop/uec-internship-2025-2nd-team2-develop/web-camera-sample/Fire_Detection-main/fire_model.pt'

#読み込み処理
pygame.mixer.init() #初期化
pygame.mixer.music.load(mp3file_path) #読み込み
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

def Sound():
    playflag = pygame.mixer.music.get_busy()    
    if(playflag == False):
        pygame.mixer.music.play(1) #ループ再生（引数を1にすると1回のみ再生）

#model = YOLO(YOLOmodel_path)
#model_fire = YOLO(Firemodel_path)

#model.export(format="openvino")  # creates 'yolo11n_openvino_model/'
# Load the exported OpenVINO model
ov_model = YOLO("yolo11n_openvino_model/")
#model_fire.export(format="openvino")
ov_model_fire = YOLO("fire_model_openvino_model/")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("cap not Open")
    exit()
maxframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

speedestimator = solutions.SpeedEstimator(
    model=ov_model_fire,
    conf = 0.2,
    classes=[0],
    show=False,
    meter_per_pixel=0.05,
    iou = 0.7,
)

def showimage():
    global video_path
    global current_index
    global cap
    global resizescale
    global label
    global maxframe
    global framecount
    if cap:
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
    
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
        
        text = f"{current_index}/{maxframe}"    
        framecount.set(text)  

        frame = cv2.resize(frame, None, fx=resizescale, fy=resizescale)
        cv2.imshow("Image Viewer", frame)
    else: print("error")


#上下キー入力(あんま意味なくて悲しい)
def on_key_press(event):
    # print("successcall")
    global current_index
    global label
    global maxframe
    if event.keysym == "Down" and current_index > 0:
        current_index = current_index - 1
        print(current_index)
    elif event.keysym == "Up" and current_index >= 0 and current_index < maxframe:
        current_index = current_index + 1
        print(current_index)
    elif event.keysym == "Up" and current_index == maxframe:
        current_index = 0
        print(current_index)
    elif event.keysym == "Down" and current_index == 0:
        current_index = current_index
        print(current_index)
    elif event.keysym == "q":
        exit()
        
    # text = f"{current_index}/{maxframe}"    
    # framecount.set(text)  
    showimage()
        

# YOLO
def run_speed_program():
    global video_path
    global ov_model
    global ov_model_fire
    global current_index
    global cap
    global dangerdistance
    global resizescale
    global waitkeytime
    global maxframe
    global framecount
    
    ret, frame = cap.read()
    ROI = cv2.selectROI('Select ROIs', cv2.resize(frame, None, fx=resizescale, fy=resizescale), fromCenter = False, showCrosshair = False)
    x_1 = ROI[0]
    y_1 = ROI[1]
    x_2 = ROI[2]
    y_2 = ROI[3]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
        
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_index = int(current_frame)
        text = f"{current_index}/{maxframe}"    
        framecount.set(text)  

        cv2.rectangle(frame,(x_1,y_1),(x_1+x_2,y_1+y_2),(0,0,255),0)

        X1 = 0
        Y1 = 0
        X2 = 0
        Y2 = 0
    
        result = ov_model_fire(frame,device="intel:gpu")
        frame = frame.copy()
        i = 0
        for info in result:
            boxes = info.boxes
            
            x1,y1,x2,y2  =0,0,0,0
            for box in boxes:
                confidence = box.conf[0]
                confidence = math.ceil(confidence * 100)
                Class = int(box.cls[0])
                if confidence > 20:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    #cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 0, 255), 5)
                    i = i + 1
                X1  =X1 + x1
                Y1 = Y1 + y1
                X2 = X2 + x2
                Y2 = Y2 + y2
        if i != 0:
            X1_av = int(X1/i +(X2-X1)/(2*i))
            Y1_av = int(Y1/i + (Y2-Y1)/(2*i))
            cv2.rectangle(frame,(X1_av,Y1_av),(X1_av+10,Y1_av +10),(0,255,0),5)
            distance = math.sqrt((x_1+(x_2/2)-X1_av)**2 +(y_1+(y_2/2)-Y1_av)**2 )
            cv2.putText(frame, f"distance:{distance}", (0, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
        results = speedestimator(frame)
        track_num = 0
        speed_val = 0
        for i in range(results.total_tracks):
            speed_val =speed_val + results.speed['track']
            track_num = track_num +1
        if track_num!=0:
            speed_ave =speed_val/track_num * 3.6 * 0.07
            cv2.putText(frame, f"speed_average:{speed_ave}", (0, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
            cv2.putText(frame, f"reach_estimate_time:{distance/speed_ave}", (0, 150), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255))
        
        frame = cv2.resize(frame, None, fx=resizescale, fy=resizescale)
        cv2.imshow("frame",frame)

        if cv2.waitKey(waitkeytime) & 0xFF == ord('q'):

            # current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # current_index = int(current_frame)
            print("break success")
            print(current_index)
            showimage()

            cv2.destroyWindow('DIstance') 
            break
        
    return

        
def run_yolo_program():
    print("run_yolo_program")
    global video_path
    global ov_model
    global ov_model_fire
    global current_index
    global cap
    global dangerdistance
    global resizescale
    global waitkeytime
    global maxframe
    global framecount
    
    righthuman = None
    #cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_index = int(current_frame)
        text = f"{current_index}/{maxframe}"    
        framecount.set(text) 

        results = ov_model.predict(frame, conf=0.8, classes=[0])
        results2 = ov_model_fire.predict(frame, conf=0.2, classes=[0])

        annotated_frame = results[0].plot()
        annotated_frame = results2[0].plot(img=annotated_frame)
        
        #cv2.namedWindow('YOLO_Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('YOLO_Detection', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)



        for box, cls in zip(results[0].boxes, results[0].boxes.cls):
            lefthuman, tophuman, righthuman, bottomhuman = [int(i) for i in box.xyxy[0]]
            # print(f"HUMAN:StartX={x1human}, StartY={y1human}, EndX={x2human}, EndY={y2human}")

        for box, cls in zip(results2[0].boxes, results2[0].boxes.cls):
            # print("fire in comming")
            leftfire, topfire, rightfire, bottomfire = [int(i) for i in box.xyxy[0]]
            cv2.rectangle(annotated_frame,(leftfire - dangerdistance,topfire - dangerdistance),(rightfire + dangerdistance,bottomfire + dangerdistance),(0,0,255),cv2.LINE_4)
            

            if righthuman:
                if(righthuman < leftfire):
                    dx = leftfire - righthuman
                elif rightfire < lefthuman:
                    dx = lefthuman - rightfire
                else:
                    dx = 0
                    
                if(bottomhuman < topfire):
                    dy = topfire - bottomhuman
                elif bottomfire < tophuman:
                    dy = tophuman - bottomfire
                else:
                    dy = 0
                    
                distance = math.sqrt(dx**2 + dy**2)
                
                # print(distance)
                cv2.putText(annotated_frame,"distance = "+str(distance),(0, 200),cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,0),thickness = 2)
                
                if distance < dangerdistance:

                    Sound()
                    cv2.putText(annotated_frame,"danger",(0, 50),cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255))
                    # current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    # current_index = int(current_frame)
                    

        #annotated_frame = cv2.resize(annotated_frame, (640,360))
        annotated_frame = cv2.resize(annotated_frame, None, fx=resizescale, fy=resizescale) 
        cv2.imshow('YOLO_Detection', annotated_frame)
        
        if cv2.waitKey(waitkeytime) & 0xFF == ord('q'):


            print("break success")
            print(current_index)

            showimage()
            cv2.destroyWindow('YOLO_Detection') 
            break

    return

def depth_estimate(img,model_kind,model_clas_id,model_conf,scale_f):
    global midas

    model = model_kind
    
    scale_factor = scale_f
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms(img_rgb)
    with torch.no_grad():
        prediction = midas(input_tensor)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map,  (img.shape[1], img.shape[0]))
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

    fx = 500.0
    fy = 500.0
    cx = img.shape[1] / 2
    cy = img.shape[0] / 2

    results = model(img,conf=model_conf)[0]
    annotated = img.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == model_clas_id:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # バウンディングボックス内の深度の中央値
            depth_roi = depth_map_norm[y1:y2, x1:x2]
            if depth_roi.size == 0:
                continue
            z_rel = np.median(depth_roi)
            z = z_rel * scale_factor  # メートルに変換

            # ピクセル座標 → カメラ座標
            px, py = (x1 + x2) // 2, (y1 + y2) // 2
            Xc = (px - cx) * z / fx
            Yc = (py - cy) * z / fy
            Zc = z

            # 表示処理
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"XYZ: ({Xc:.2f}, {Yc:.2f}, {Zc:.2f})", (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # コンソール出力
            # print(f"Person bbox: ({x1},{y1})-({x2},{y2})")
            # print(f"Depth (Z): {Zc:.3f} ")
            # print(f"Camera Coords (X, Y, Z): ({Xc:.3f}, {Yc:.3f}, {Zc:.3f})/n")
    return annotated

def run_distance_program():
    print("run_distance_program")
    global video_path
    global ov_model
    global ov_model_fire
    global current_index
    global cap
    global dangerdistance
    global resizescale
    global waitkeytime

    #cap.set(cv2.CAP_PROP_POS_FRAMES, current_index)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
                
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_index = int(current_frame)
        text = f"{current_index}/{maxframe}"    
        framecount.set(text) 

        results = ov_model.predict(frame, conf=0.8, classes=[0])
        results2 = ov_model_fire.predict(frame, conf=0.2, classes=[0])

        annotated_frame = results[0].plot()
        annotated_frame = results2[0].plot(img=annotated_frame)
        
        #cv2.namedWindow('YOLO_Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Distance', cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
        annotated_frame2 = depth_estimate(annotated_frame,ov_model_fire,0,0.2,3.0)                    
        #annotated_frame = cv2.resize(annotated_frame2, (640,360))
        annotated_frame = cv2.resize(annotated_frame2, None, fx=resizescale, fy=resizescale)
        
        
        cv2.imshow('Distance', annotated_frame)

        
        if cv2.waitKey(waitkeytime) & 0xFF == ord('q'):

            # current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # current_index = int(current_frame)
            print("break success")
            print(current_index)
            showimage()

            cv2.destroyWindow('DIstance') 
            break
        

    return

root = tk.Tk()
root.title("フレーム選択")

framecount = tk.StringVar()
framecount.set(f"{current_index}/{maxframe}")

label = tk.Label(
    root,
    textvariable=framecount
)
label.pack()

def run_exit():
    exit


root.bind("<Up>", on_key_press)
root.bind("<Down>", on_key_press)
root.bind("<KeyPress-q>", on_key_press)

if App_start_Flag:
    showimage()
    App_start_Flag = False

#YOLO起動ボタン
def start_yolo():
    threading.Thread(target=run_yolo_program, daemon=True).start()
def start_speed():
    threading.Thread(target=run_speed_program, daemon=False).start()
def start_3dDistance():
    threading.Thread(target=run_distance_program, daemon=False).start()

def fire_depth_estimate(img,model_clas_id,model_conf,scale_f):
    
    scale_factor = scale_f
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms(img_rgb)
    with torch.no_grad():
        
        prediction = midas(input_tensor)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = cv2.resize(depth_map,  (img.shape[1], img.shape[0]))
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)

    fx = 500.0
    fy = 500.0
    cx = img.shape[1] / 2
    cy = img.shape[0] / 2

    results = ov_model_fire(img,conf=model_conf,device="intel:gpu")[0]
    annotated = img.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == model_clas_id:  # person
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # バウンディングボックス内の深度の中央値
            depth_roi = depth_map_norm[y1:y2, x1:x2]
            if depth_roi.size == 0:
                continue
            z_rel = np.median(depth_roi)
            z = z_rel * scale_factor  # メートルに変換

            # ピクセル座標 → カメラ座標
            px, py = (x1 + x2) // 2, (y1 + y2) // 2
            Xc = (px - cx) * z / fx
            Yc = (py - cy) * z / fy
            Zc = z

            # 表示処理
            #cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"XYZ: ({Xc:.2f}, {Yc:.2f}, {Zc:.2f}) ", (x1, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # コンソール出力
            #print(f"Person bbox: ({x1},{y1})-({x2},{y2})")
            #print(f"Depth (Z): {Zc:.3f} m")
            #print(f"Camera Coords (X, Y, Z): ({Xc:.3f}, {Yc:.3f}, {Zc:.3f})/n")
    return annotated

btn = tk.Button(root, text="start", command=start_yolo)
btn1 = tk.Button(root, text="depth", command=start_3dDistance)
btn2 = tk.Button(root, text="speed", command=start_speed)
btn.pack(pady=10)
btn1.pack(pady=10)
btn2.pack(pady=10)

root.mainloop()

# if cv2.waitKey(waitkeytime) & 0xFF == ord('q'):
#     cv2.destroyAllWindows()