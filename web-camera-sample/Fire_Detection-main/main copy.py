import cv2
from ultralytics import YOLO

# 動画ファイルのパス
video_path = 'syouka.mp4'

# Yolov8モデルのロード
model = YOLO('yolov8n.pt')
model2 = YOLO('fire_model.pt')

# 動画ファイルの読み込み
cap = cv2.VideoCapture(video_path)

# 出力動画の設定
output_path = 'test.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()
    scaleparameter = 0.4
    frame = cv2.resize(frame, None, fx=scaleparameter, fy=scaleparameter)
    if ret:
        # フレームごとに物体検知を行う
        results = model(frame)
        results2 = model2(frame)
        
        # 検知結果を描画
        annotated_frame = results[0].plot()
        annotated_frame2 = results2[0].plot()
        
        # 出力動画にフレームを書き込む
        #out.write(annotated_frame)
        
        # フレームを表示
        im_h = cv2.hconcat([annotated_frame, annotated_frame2])
        cv2.imshow('Frame', im_h)
        #cv2.imshow('Frame', annotated_frame)
        #cv2.imshow('Frame', annotated_frame2)
        
        # 'q'キーが押されたら終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# リソースの解放
cap.release()
out.release()
cv2.destroyAllWindows()