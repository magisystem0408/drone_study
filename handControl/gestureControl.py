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


def getLength(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def calcAngle(right_x, right_y, middle_x, middle_y, top_x, top_y):
    radians = np.arctan2(top_y - middle_y, top_x - middle_x) - np.arctan2(right_y - middle_y, right_x - middle_x)
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle

    return angle


while True:
    success, img = cap.read()
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # 親指
        right_thumb_tip_x, right_thumb_tip_y = lmList[4][1], lmList[4][2]
        # 人差し指
        right_index_finger_tip_x, right_index_finger_tip_y = lmList[8][1], lmList[8][2]
        # 人差しの付け根より一つ上
        right_index_finger_pip_x, right_index_finger_pip_y = lmList[6][1], lmList[6][2]
        # 人差しの付け根より上
        right_index_finger_mcp_x, right_index_finger_mcp_y = lmList[5][1], lmList[5][2]

        # 中指
        right_middle_finger_tip_x, right_middle_finger_tip_y = lmList[12][1], lmList[12][2]
        # 中指の付け根より一つ上
        right_middle_finger_pip_x, right_middle_finger_pip_y = lmList[10][1], lmList[10][2]
        # 中指の付け根
        right_middle_finger_mcp_x, right_middle_finger_mcp_y = lmList[9][1], lmList[9][2]

        # 薬指
        right_ring_finger_tip_x, right_ring_finger_tip_y = lmList[16][1], lmList[16][2]
        # 小指
        right_pinky_tip_x, right_pinky_tip_y = lmList[20][1], lmList[20][2]

        # cv2.line(img, (right_thumb_tip_x, right_thumb_tip_y), (right_ring_finger_tip_x, right_ring_finger_tip_y),
        #          (255, 0, 255), 3)

        cv2.line(img, (right_thumb_tip_x, right_thumb_tip_y), (right_pinky_tip_x, right_pinky_tip_y),
                 (255, 0, 255), 3)

        # チョキ判定
        thum_ring_length = getLength(right_thumb_tip_x, right_thumb_tip_y,
                                     right_ring_finger_tip_x, right_ring_finger_tip_y
                                     )

        thum_pinky_length = getLength(right_thumb_tip_x, right_thumb_tip_y,
                                      right_pinky_tip_x, right_pinky_tip_y,
                                      )
        # 角度計算
        index_angle = calcAngle(right_index_finger_mcp_x, right_index_finger_mcp_y,
                                right_index_finger_pip_x, right_index_finger_pip_y,
                                right_index_finger_tip_x, right_index_finger_tip_y
                                )

        middle_angle =calcAngle(right_middle_finger_mcp_x,right_middle_finger_mcp_y,
                                right_middle_finger_pip_x,right_middle_finger_pip_y,
                                right_middle_finger_tip_x,right_middle_finger_tip_y
                                )

        if thum_pinky_length <= 40 and thum_ring_length <= 20 \
                and index_angle >= 170 and middle_angle>=170:
            print("これはチョキです。")

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("img", img)
    cv2.waitKey(1)
