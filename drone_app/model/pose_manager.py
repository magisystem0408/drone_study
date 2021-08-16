import mediapipe as mp
import cv2
import numpy as np

# 640*480

FRAME_X = 640
FRAME_Y = 480
from pose_estimation_class import PoseDetector as pec

class MediaPose():
    def __init__(self, mp_drawing=mp.solutions.drawing_utils,
                 mp_pose=mp.solutions.pose,
                 cap=cv2.VideoCapture(2)
                 ):

        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.cap = cap

        self.start()

    def start(self):

        def calculate_angle(first, middle, end):
            first = np.array(first)
            middle = np.array(middle)
            end = np.array(end)

            radians = np.arctan2(end[1] - middle[1], end[0] - middle[0]) - np.arctan2(first[1] - middle[1],
                                                                                      first[0] - middle[0])
            angle = np.abs(radians * 180 / np.pi)
            if angle > 180.0:
                angle = 360 - angle
            return angle

        def draw_text(image, angle, middle_joint):
            cv2.putText(image, str(angle),
                        tuple(np.multiply(middle_joint, [FRAME_X, FRAME_Y]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                        )


        with self.mp_pose.Pose(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5
                               ) as pose:

            while self.cap.isOpened():
                ret, frame = self.cap.read()

                frame = cv2.resize(frame, dsize=(FRAME_X, FRAME_Y))

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                try:
                    landmarks = results.pose_landmarks.landmark

                    # 左
                    left_sholder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                                    ]

                    left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                                  ]

                    left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                  landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
                                  ]

                    left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
                                ]

                    # 右
                    right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                      landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                    right_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

                    right_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                   landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y
                                   ]

                    right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y
                                 ]

                    left_arm_angle =pec.calculate_angle(left_sholder, left_elbow, left_wrist)
                    right_arm_angle = pec.calculate_angle(right_shoulder, right_elbow, right_wrist)
                    left_body_angle = pec.calculate_angle(left_hip, left_sholder, left_elbow)
                    right_body_angle = pec.calculate_angle(right_hip, right_shoulder, right_elbow)

                    cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)

                    draw_text(image, left_arm_angle, left_elbow)
                    draw_text(image, right_arm_angle, right_elbow)
                    draw_text(image, left_body_angle, left_sholder)
                    draw_text(image, right_body_angle, right_shoulder)

                    pose_detect =pec(left_arm_angle, right_arm_angle, left_body_angle, right_body_angle)
                    detect_a =pose_detect.pose_A()
                    detect_l =pose_detect.pose_L()
                    detect_t =pose_detect.pose_T()


                    cv2.putText(image, str('pose:A') + str(detect_a), (10, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )

                    cv2.putText(image, str('poseL:') + str(detect_l), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                                )

                    cv2.putText(image, str('poseT:') + str(detect_t), (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
                                )
                    # if pose_t:
                    #     return print('test')

                except:
                    pass

                self.mp_drawing.draw_landmarks(image,
                                               results.pose_landmarks,
                                               self.mp_pose.POSE_CONNECTIONS,
                                               self.mp_drawing.DrawingSpec(color=(245, 117, 66),
                                                                           thickness=2,
                                                                           circle_radius=10,
                                                                           ),
                                               self.mp_drawing.DrawingSpec(color=(245, 117, 66),
                                                                           thickness=2,
                                                                           circle_radius=10,
                                                                           ),
                                               )
                cv2.imshow('test', image)
                if cv2.waitKey(10) and 0xFF == ord('q'):
                    break

        self.cap.release()
        self.cap.destroyAllWindows()


if __name__ == '__main__':
    MediaPose()
