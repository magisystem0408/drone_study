import cv2


# アニメ風エフェクト
def animationEffect(imgfile: str):
    color_orig_image = cv2.imread(imgfile, cv2.IMREAD_COLOR)

    h, w = color_orig_image.shape[:2]
    #   color_orig_image = cv2.resize(color_orig_image, (w*2,h*2))
    numDownSamples = 2  # スケーリング数
    numBilateralFilters = 20  # バイラテラルフィルタ適用回数

    #############################################
    # 減色工程

    # ガウシアンピラミッドによるダウンサンプリング
    img_color = color_orig_image
    for _ in range(numDownSamples):
        img_color = cv2.pyrDown(img_color)

        # バイラテラルフィルターを繰り返し適用する
    for _ in range(numBilateralFilters):
        img_color = cv2.bilateralFilter(img_color, 7, 7, 7)

        # アップサイジングでオリジナルサイズまで戻す
    for _ in range(numDownSamples):
        img_color = cv2.pyrUp(img_color)

        #############################################
    # エッジ取得工程

    # グレイスケール化とガウシアンフィルタ
    img_gray = cv2.cvtColor(color_orig_image, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.medianBlur(img_gray, 5)

    # 二値化 エッジ取得
    img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    #############################################
    # エッジイメージと減色イメージをANDで合成
    h, w = img_color.shape[:2]
    img_edge = cv2.resize(img_edge, (w, h))

    # 3チャンネル化
    img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(img_color, img_edge)


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    while True:
        _, img = cap.read()

        height, width = img.shape[:2]

        numDownSamples = 1  # スケーリング数
        numBilateralFilters = 9  # バイラテラルフィルタ適用回数

        # 減色
        img_color = img
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 7, 7, 7)

            # アップサイジングでオリジナルサイズまで戻す
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 5)


        # 二値化 エッジ取得
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        h, w = img_color.shape[:2]
        img_edge = cv2.resize(img_edge, (w, h))

        # 3チャンネル化
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)


        img = cv2.bitwise_and(img_color, img_edge)




        cv2.imshow("image", img)

        cv2.waitKey(1)
        cv2.destroyAllWindows()
