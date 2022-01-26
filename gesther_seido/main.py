import threading
import cv2
import time
import math
import glob
import numpy as np
import osascript
import pygame
import handDetector as htm


class HandControl(object):
    def __init__(self, blob_ip: str = "192.168.10.05"):
        self.cap = cv2.VideoCapture(1)
        self.detector = htm.handDetector()
        self.main_mode = True
        self.music_mode = False
        self.music_play_mode = True
        self.volumeFlag = False
        self.islighting = False
        self.crashSize = 300
        self._command_semaphore = threading.Semaphore(1)
        self._command_thread = None
        self.gesture()

    def _getLength(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def _calcAngle(self, right_x, right_y, middle_x, middle_y, top_x, top_y):
        radians = np.arctan2(top_y - middle_y, top_x - middle_x) - np.arctan2(right_y - middle_y, right_x - middle_x)
        angle = np.abs(radians * 180 / np.pi)
        if angle > 180:
            angle = 360 - angle
        return angle


    def gesture(self):
        while True:
            success, img = self.cap.read()

            img = self.detector.findHand(img,draw=False)

            lmList = self.detector.findPosition(img, draw=False)
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

                # 薬指
                right_ring_finger_tip_x, right_ring_finger_tip_y = lmList[16][1], lmList[16][2]

                # 小指
                right_pinky_tip_x, right_pinky_tip_y = lmList[20][1], lmList[20][2]

                # 長さ判定
                thumb_middle_length = self._getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                      right_middle_finger_tip_x, right_middle_finger_tip_y)
                thumb_ring_length = self._getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                    right_ring_finger_tip_x, right_ring_finger_tip_y)

                thumb_pinky_length = self._getLength(right_thumb_tip_x, right_thumb_tip_y,
                                                     right_pinky_tip_x, right_pinky_tip_y, )

                # 角度計算(大回りよう)


                index_angle = self._calcAngle(right_index_finger_mcp_x, right_index_finger_mcp_y,
                                              right_index_finger_pip_x, right_index_finger_pip_y,
                                              right_index_finger_tip_x, right_index_finger_tip_y
                                              )




                firstGeture = index_angle >= 170 and thumb_ring_length <= 20 and thumb_middle_length <= 55 and thumb_pinky_length <= 50

                if firstGeture:
                    avg = None
                    # # グレースケールに変換
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # 比較用のフレームを取得する
                    if avg is None:
                        avg = gray.copy().astype("float")
                        continue
                    #
                    # # 現在のフレームと移動平均との差を計算
                    cv2.accumulateWeighted(gray, avg, 0.97)
                    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
                    #
                    # # デルタ画像を閾値処理を行う
                    thresh = cv2.threshold(frameDelta, 1.5, 255, cv2.THRESH_BINARY)[1]

                    # 画像の閾値に輪郭線を入れる
                    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (0, 0, 255), 10)
                    cv2.drawContours(img, contours, -1, (255, 255, 255), 1)
                    cv2.drawContours(img, contours, -1, (0, 255, 255), 4)


                    cv2.putText(img, "1", (right_index_finger_tip_x, right_index_finger_tip_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4, cv2.LINE_AA)




            cv2.imshow("img", img)
            cv2.waitKey(1)


if __name__ == '__main__':
    controller = HandControl()
