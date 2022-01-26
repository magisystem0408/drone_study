import cv2
import time
import math
import numpy as np
import handDetector as htm

# wCam, hCam = 640, 480

cap = cv2.VideoCapture(1)
# cap.set(3, wCam)
# cap.set(4, hCam)

pTime = 0

detector = htm.handDetector()



while True:
    success, img = cap.read()
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255,), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)


        # 親指と人差し指の距離の長さを取得
        length = math.hypot(x2 - x1, y2 - y1)

        # 0~100のスケールに正規化する
        norm_length=np.interp(length, [20, 300], [0, 100])



        # ここにコマンド処理を書いていく
        # if length < 50:
        #     cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)



    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)
