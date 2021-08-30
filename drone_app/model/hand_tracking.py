import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot as plt
import uuid
import os

mp_drawing = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands


# 検知する
def get_label(index, hand, results):
    output = None
    for idx, classification in enumerate(results.multi_handedness):
        if classification.classification[0].index == index:
            label = classification.classification[0].label
            score = classification.classification[0].score
            text = '{}{}'.format(label, round(score, 2))

            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hand.HandLandmark.WRIST].x,
                          hand.landmark[mp_hand.HandLandmark.WRIST].y,)),
                [640, 480]).astype(int))
            output = text, coords

    return output


def get_et(index, hand, results):
    output = None

    for idx, classification in enumerate(results.multi_handedness):
        # 左のパターンと右のパターンで判定する
        if classification.classification[0].index == index:
            # 左か右か
            label = classification.classification[0].label
            # その座標
            score = classification.classification[0].score

            text = '{}{}'.format(label, round(score, 2))

            coords = tuple(np.multiply(
                np.array((hand.landmark[mp_hand.HandLandmark.INDEX_FINGER_TIP].x,
                          hand.landmark[mp_hand.HandLandmark.INDEX_FINGER_TIP].y,)),
                [640, 480]).astype(int))

            # hand[index] =coords
            output = text, coords,label
    return output



# 一緒に座標が一致された時にTrueになる
# 2点間の距離が測定される

def detect_t():
    output=None
    # if




joint_list = [[8, 7, 6], [12, 11, 10]]


def draw_finger_angles(image, results, joint_list):
    for hand in results.multi_hand_landmarks:
        for joint in joint_list:

            # それぞれの関節が入る
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y])
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

            # ここから角度計算が入る
            radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            angle = np.abs(radians * 180.0 / np.pi)

            if angle > 180:
                angle = 360 - angle
            cv2.putText(image, str(round(angle, 2)), tuple(np.multiply(b, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return image




cap = cv2.VideoCapture(2)
with mp_hand.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, dsize=(640, 480))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = cv2.flip(image, 1)
        image.flags.writeable = False

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hand.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          )

                # if get_label(num, hand, results):
                #     text, coord = get_label(num, hand, results)
                #     cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # right_x =0
                # right_y=0
                # left_x=0
                # left_y =0

                if get_et(num,hand,results):
                    text, coord,label = get_et(num, hand, results)

                    if label=='Right':
                        right_x,right_y =coord
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

                    if label=='Left':
                        # アンパッキング
                        left_x,left_y=coord
                        cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


                # image =draw_finger_angles(image,results,joint_list)

        cv2.imshow("hand", image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destoryAllWindows()
