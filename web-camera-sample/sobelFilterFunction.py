import cv2 # need to import extra module "pip install opencv-python"
import numpy as np

#引数：img 2次元配列を出力してください。3次元配列だった場合、グレースケールに変換されます
#出力：img 3次元配列が返されます

def sobelFilter(img):
    kernel_size = 5                                                         # カーネルサイズの設定
    sigma = 0                                                               # sigmaの設定
    
    #if(img.ndim == 2):
    #    img = np.stack((img,)*3, -1)
    if(img.ndim == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)    # ガウシアンフィルターの適用


    # Sobelフィルタを適用
    sobel_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3,scale=3)                 # 水平方向の勾配
    sobel_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3,scale=3)                 # 垂直方向の勾配

    # # 勾配の絶対値を計算
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)

     # 水平方向と垂直方向の勾配を組み合わせて合成勾配を計算
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    #
    sobel_combined_red = np.stack((sobel_combined,)*3, -1) #BGR
    sobel_combined_red[:,:,0] = 0
    sobel_combined_red[:,:,1] = 0

    # マスク画像生成のためだけに色反転画像(negativeimage)を作成
    negative_image = cv2.bitwise_not(sobel_combined_red)
    # 白色部分に対応するマスク画像を生成
    mask = np.all(negative_image[:,:,:] > [200, 200, 200], axis=-1)
    
    # マスク画像をBGR形式からBGRA形式に変換
    dst = cv2.cvtColor(negative_image, cv2.COLOR_BGR2BGRA)
    # 元画像をBGR形式からBGRA形式に変換
    image_alpha = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    #マスク画像の座標と一致する部分のアルファチャンネルを０に
    dst[mask,3] = 0

    #アルファチャンネルつき元画像にマスク処理されたsobelフィルタイメージを重ねる
    filteredImage_4dim = image_alpha+dst

    #アルファチャンネルなしにもどして出力
    filteredImage_3dim = filteredImage_4dim[:, :, 0:3]

    return sobel_combined_red