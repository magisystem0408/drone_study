import cv2
import matplotlib.pyplot as plt
import handDetector as htm
import numpy as np
import math

hand = htm.handDetector()


def _getLength(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def _calcAngle(right_x, right_y, middle_x, middle_y, top_x, top_y):
    radians = np.arctan2(top_y - middle_y, top_x - middle_x) - np.arctan2(right_y - middle_y, right_x - middle_x)
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


# neko.jpgを読み込んで、imgオブジェクトに入れる
img = cv2.imread("images/one/one_01.jpg")
img = hand.findHand(img)

lmList = hand.findPosition(img)

if len(lmList) != 0:
    # 親指
    right_thumb_tip_x, right_thumb_tip_y = lmList[4][1], lmList[4][2]
    # 親指のした
    right_thumb_ip_x, right_thum_ip_y = lmList[3][1], lmList[3][2]
    # 更にした
    right_thumb_cmc_x, right_thum_cmc_y = lmList[2][1], lmList[2][1]

    # 人差し指
    right_index_finger_tip_x, right_index_finger_tip_y = lmList[8][1], lmList[8][2]
    # 人差し指のした
    right_index_finger_dip_x, right_index_finger_dip_y = lmList[7][1], lmList[7][2]
    # 人差しの付け根より一つ上
    right_index_finger_pip_x, right_index_finger_pip_y = lmList[6][1], lmList[6][2]
    # 人差しの付け根より上
    right_index_finger_mcp_x, right_index_finger_mcp_y = lmList[5][1], lmList[5][2]

    # 中指
    right_middle_finger_tip_x, right_middle_finger_tip_y = lmList[12][1], lmList[12][2]
    # 中指のした
    right_middle_finger_dip_x, right_middle_finger_dip_y = lmList[11][1], lmList[11][2]
    # 中指の付け根より一つ上
    right_middle_finger_pip_x, right_middle_finger_pip_y = lmList[10][1], lmList[10][2]
    # 中指の付け根
    right_middle_finger_mcp_x, right_middle_finger_mcp_y = lmList[9][1], lmList[9][2]

    # 薬指
    right_ring_finger_tip_x, right_ring_finger_tip_y = lmList[16][1], lmList[16][2]
    # 薬指のした
    right_ring_finger_dip_x, right_ring_finger_dip_y = lmList[15][1], lmList[15][2]
    # 薬指の付け根のしたより一つ上
    right_ring_finger_pip_x, right_ring_finger_pip_y = lmList[14][1], lmList[14][2]
    # 薬指の付け根
    right_ring_finger_mcp_x, right_ring_finger_mcp_y = lmList[13][1], lmList[13][2]

    # 小指
    right_pinky_tip_x, right_pinky_tip_y = lmList[20][1], lmList[20][2]
    # 小指のした
    right_pinky_dip_x, right_pinky_dip_y = lmList[19][1], lmList[19][2]
    # 小指の付け根より一つ上
    right_pinky_pip_x, right_pinky_pip_y = lmList[18][1], lmList[18][2]
    # 小指の付け根
    right_pinky_mcp_x, right_pinky_mcp_y = lmList[17][1], lmList[17][2]

    thumb_index_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                    right_index_finger_tip_x, right_index_finger_tip_y)
    thumb_middle_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                     right_middle_finger_tip_x, right_middle_finger_tip_y)

    thumb_middle_length_pip = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                         right_middle_finger_pip_x, right_middle_finger_pip_y
                                         )
    thumb_middle_length_dip = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                         right_middle_finger_dip_x, right_middle_finger_dip_y
                                         )

    thumb_ring_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                   right_ring_finger_tip_x, right_ring_finger_tip_y)

    thumb_pinky_length = _getLength(right_thumb_tip_x, right_thumb_tip_y,
                                    right_pinky_tip_x, right_pinky_tip_y, )

    # 角度計算(大回りよう)
    thumb_angle = _calcAngle(right_thumb_cmc_x, right_thum_cmc_y,
                             right_thumb_ip_x, right_thum_ip_y,
                             right_thumb_tip_x, right_thumb_tip_y
                             )

    index_angle = _calcAngle(right_index_finger_mcp_x, right_index_finger_mcp_y,
                             right_index_finger_pip_x, right_index_finger_pip_y,
                             right_index_finger_tip_x, right_index_finger_tip_y
                             )

    middle_angle = _calcAngle(right_middle_finger_mcp_x, right_middle_finger_mcp_y,
                              right_middle_finger_pip_x, right_middle_finger_pip_y,
                              right_middle_finger_tip_x, right_middle_finger_tip_y
                              )
    ring_angle = _calcAngle(right_ring_finger_mcp_x, right_ring_finger_mcp_y,
                            right_ring_finger_pip_x, right_ring_finger_pip_y,
                            right_ring_finger_tip_x, right_ring_finger_tip_y,
                            )

    pinky_angle = _calcAngle(right_pinky_mcp_x, right_pinky_mcp_y,
                             right_pinky_pip_x, right_pinky_pip_y,
                             right_pinky_tip_x, right_pinky_tip_y
                             )

    # 角度計算(小回り用)
    index_angle_2 = _calcAngle(right_index_finger_pip_x, right_index_finger_pip_y,
                               right_index_finger_dip_x, right_middle_finger_dip_y,
                               right_index_finger_tip_x, right_index_finger_tip_y,
                               )

    middle_angle_2 = _calcAngle(right_middle_finger_pip_x, right_middle_finger_pip_y,
                                right_middle_finger_dip_x, right_middle_finger_dip_y,
                                right_middle_finger_tip_x, right_middle_finger_tip_y
                                )
    ring_angle_2 = _calcAngle(right_ring_finger_pip_x, right_ring_finger_pip_y,
                              right_ring_finger_dip_x, right_ring_finger_dip_y,
                              right_ring_finger_tip_x, right_ring_finger_tip_y
                              )
    pinky_angle_2 = _calcAngle(right_pinky_pip_x, right_pinky_tip_y,
                               right_pinky_dip_x, right_pinky_dip_y,
                               right_pinky_tip_x, right_pinky_tip_y
                               )

    firstGeture = index_angle >= 170 and (
            thumb_middle_length <= 60 or thumb_middle_length_pip <= 65 or thumb_middle_length_dip <= 50)

    


    secondGesture = thumb_pinky_length <= 40 and thumb_ring_length <= 20 and index_angle >= 170 and middle_angle >= 170
    threeGesture = index_angle >= 170 and middle_angle >= 170 and ring_angle >= 170 and thumb_pinky_length <= 40
    yeyGeture = index_angle >= 170 and pinky_angle >= 170 and thumb_ring_length <= 40 and thumb_middle_length <= 40

    if firstGeture:
        cv2.putText(img, "2", (right_index_finger_tip_x, right_index_finger_tip_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 4, cv2.LINE_AA)
        print("認識確認できました。")

else:
    print("認識できませんでした。")

# 画像の色の順序をBGRからRGBに変換する
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# imgオブジェクトをmatlotlibを用いて表示する
plt.imshow(img)
plt.show()
