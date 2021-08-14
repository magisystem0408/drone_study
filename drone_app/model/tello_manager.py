import socket
import threading
import time
import cv2

import os
import subprocess
import numpy as np

import mediapipe as mp

# ドローンのデフォルトでは960*720で送られてくる

FRAME_X = int(960 / 3)
FRAME_Y = int(720 / 3)
FRAME_AREA = FRAME_X * FRAME_Y

# ffmpegで多くの情報を処理するため
FRAME_SIZE = FRAME_AREA * 3

# CMD_FFMPEG =f'ffmpeg -hwaccel auto -hwaccel_device opencl -i pipe:0 -pix_fmt bgr24 -s {FRAME_X}x{FRAME_Y} -f rawvideo pipe:1'


FRAME_POSE_X = 640
FRAME_POSE_Y = 480

FACE_DETECT_XML_FILE = '/Users/matsudomasato/PycharmProjects/studydrone/drone_app/model/haarcascade_frontalface_default.xml'
EYE_DETECT_XML_FILE = '/Users/matsudomasato/PycharmProjects/studydrone/drone_app/model/haarcascade_eye.xml'


class Tello():
    def __init__(self, host_ip: object = '192.168.10.2', host_port: object = 8889
                 , drone_ip: object = '192.168.10.1', drone_port: object = 8889,
                 mp_drawing: object = mp.solutions.drawing_utils,
                 mp_pose: object = mp.solutions.pose,
                 ):

        # PC(ローカル側)の情報
        self.host_ip = host_ip
        self.host_port = host_port
        # socket(ローカルの情報送信)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.host_ip, host_port))

        # ドローンの情報
        self.drone_ip = drone_ip
        self.drone_port = drone_port
        # socket
        self.drone_address = (self.drone_ip, self.drone_port)

        self.video_port = 11111

        self.response = None

        # eventはsetが押されてから他のスレッドが動き始める
        # self.stop_event=threading.Event()

        # データ受け取り用
        # スレッドでサブスレッド開く
        self._receive_video_thread = threading.Thread(
            target=self.receive_data, args=())
        # ↑これをスタートさせる

        # demonで待たずにやってもいい
        self._receive_video_thread.daemon = True
        # ビデオスレッドスタートさせる
        self._receive_video_thread.start()

        # 最初に送らなければいけない
        self.send_command('command')
        # カメラコマンド
        self.send_command('streamon')

        # self.proc =subprocess.Popen(CMD_FFMPEG.split(' '),
        #                             stdin=subprocess.PIPE,
        #                             stdout=subprocess.PIPE
        #                             )
        # # 入力
        # self.proc_stdin =self.proc.stdin
        # # 出力
        # self.proc_stdout=self.proc.stdout
        #
        # #スレッド定義
        # self._receive_video2_thread =threading.Thread(
        #     target=self.receive_video_test,
        #     args =(self.proc.stdin,self.host_ip,self.video_port)
        # )
        # スレッドスタートさせる
        # self._receive_video2_thread.start()

        # self.send_command('takeoff')

        # ビデオに送る回線コマンド
        self.cap = None
        self.video_addr = 'udp://@' + self.host_ip + ':' + str(self.video_port)

        self.face_cascade = cv2.CascadeClassifier(FACE_DETECT_XML_FILE)
        self.eye_cascade = cv2.CascadeClassifier(EYE_DETECT_XML_FILE)

        # self.video()

        self.mp_drawing = mp_drawing
        self.mp_pose = mp_pose

        self.pose()

    def send_command(self, command):
        # データ送信
        self.socket.sendto(command.encode('utf-8'), self.drone_address)

    def receive_data(self):
        while True:
            try:
                self.response, _ = self.socket.recvfrom(2048)
            except Exception as e:
                print('エラーじゃ！')
                print(e)
                # break

    def video(self):

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_addr)

        elif not self.cap.isOpend():
            self.cap.open(self.video_addr)

        while self.cap.isOpend():
            ret, frame = self.cap.read()

            frame = cv2.resize(frame, dsize=(640, 480))

            # frame,p_landmarks, p_connections=detector.findPose(frame,False)
            # mp.solutions.drawing_utils.draw_landmarks(frame, p_landmarks, p_connections)
            # lmList = detector.getPosition(frame)

            # ここから＝＝＝＝＝＝＝＝＝＝＝＝＝顔検出

            # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # faces = face.detectMultiScale(grey, 1.3, 5)
            # print(len(faces))
            # # # facesの画像の座標の高さを取得できる
            # # # wとhは幅と高さ
            # # for (x, y, w, h) in faces:
            # #     cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # #     eye_gray = grey[y:y + h, x:x + w]
            # #     eye_color = frame[y:y + h, x:x + w]
            # #     eyes = eye.detectMultiScale(eye_gray)
            # #     for (ex, ey, ew, eh) in eyes:
            # #         cv2.rectangle(eye_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            # #
            # # # ここまで＝＝＝＝＝＝＝＝＝＝＝＝＝顔検出

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# def receive_video_test(self,pipe_in,host_ip,video_port):
#     with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock_video:
#         # ソケットが空いた時にもう一度再利用する
#         sock_video.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
#
#         sock_video.bind((host_ip,video_port))
#         data =bytearray(2048)
#         while True:
#             try:
#                 # データ受け取り
#                 size,addr =sock_video.recvfrom_into(data)
#                 # print(data)
#
#             except socket.error as ex:
#                 print(ex)
#                 break
#
#             try:
#                 pipe_in.write(data[:size])
#                 # 実際にはまだ出力されていないのを出力させる
#                 pipe_in.flush()
#             except Exception as ex:
#                 print(ex)
#                 break
#
# def video_binary_generator(self):
#     while True:
#         try:
#             frame =self.proc_stdout.read(FRAME_SIZE)
#
#         except Exception as ex:
#             print(ex)
#
#         if not frame:
#             continue
#
#         frame =np.fromstring(frame,np.uint8).reshape(FRAME_X,FRAME_Y,3)
#         yield frame
#
#     for neko in frame:
#         _,jpeg =cv2.imencode('.jpg',neko)
#         cv2.imshow(jpeg)


    def takeoff(self):
        return self.send_command('takeoff')


    def land(self):
        return self.send_command('land')


    def flip_back(self):
        return self.send_command('flip b')


    def pose(self):
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

        def draw_joint_text(image, angle, middle_joint):
            cv2.putText(image, str(angle),
                        tuple(np.multiply(middle_joint, [FRAME_X, FRAME_Y]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA
                        )

        def pose_T(left_arm_angle, right_arm_angle, left_body_angle, right_body_angle):
            output = False
            if left_arm_angle >= 150 and left_body_angle >= 70 and right_arm_angle >= 150 and right_body_angle:
                return True
            return False

        def pose_A(left_body_angle, right_body_angle):
            output = False
            if left_body_angle >= 90 and right_body_angle >= 90:
                return True
            return False

        state =False
        cap = cv2.VideoCapture(0)

        with self.mp_pose.Pose(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5
                               ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()

                frame = cv2.resize(frame, dsize=(FRAME_POSE_X, FRAME_POSE_Y))

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

                    left_arm_angle = calculate_angle(left_sholder, left_elbow, left_wrist)
                    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

                    left_body_angle = calculate_angle(left_hip, left_sholder, left_elbow)
                    right_body_angle = calculate_angle(right_hip, right_shoulder, right_elbow)

                    # cv2.rectangle(image, (0, 0), (255, 73), (245, 117, 16), -1)

                    draw_joint_text(image, left_arm_angle, left_elbow)
                    draw_joint_text(image, right_arm_angle, right_elbow)
                    draw_joint_text(image, left_body_angle, left_sholder)
                    draw_joint_text(image, right_body_angle, right_shoulder)

                    pose_t = pose_T(left_arm_angle, right_arm_angle, left_body_angle, right_body_angle)
                    pose_a = pose_A(left_body_angle, right_body_angle)

                    if pose_t and state is False:
                        state =True
                        self.send_command('takeoff')

                    if pose_a and state is not False:
                        self.send_command('land')
                        state =False

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

        cap.release()
        cap.destroyAllWindows()


if __name__ == '__main__':
    tello = Tello()
    # tello.takeoff()
    # time.sleep(5)
    # tello.flip_back()
    # tello.land()

    # tello.video_binary_generator()