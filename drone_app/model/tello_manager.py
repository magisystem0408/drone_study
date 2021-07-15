import socket
import threading
import time
import cv2


import pose_estimation_class as pm
import mediapipe as mp


FRAME_X =int(960/3)
FRAME_Y =int(720/3)
FRAME_AREA =FRAME_X*FRAME_Y

FACE_DETECT_XML_FILE='/Users/matsudomasato/PycharmProjects/studydrone/drone_app/model/haarcascade_frontalface_default.xml'
EYE_DETECT_XML_FILE='/Users/matsudomasato/PycharmProjects/studydrone/drone_app/model/haarcascade_eye.xml'

class Tello():
    def __init__(self,host_ip: object = '192.168.10.2', host_port: object = 8889
                 ,drone_ip:object='192.168.10.1',drone_port:object=8889):

        # super().__init__()

        # こちらの情報
        self.host_ip =host_ip
        self.host_port=host_port
        #socket
        self.socket =socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind((self.host_ip,host_port))

        #ドローンの情報
        self.drone_ip = drone_ip
        self.drone_port =drone_port
        #socket
        self.drone_address =(self.drone_ip,self.drone_port)


        self.video_port=11111

        self.response =None
        # self.stop_event=threading.Event()

        # データ受け取り用
        self._receive_video_thread=threading.Thread(
            target=self.receive_data,args=())
        #↑これをスタートさせる
        # self._receive_video_thread.daemon =True
        self._receive_video_thread.start()


        # コマンド
        self.send_command('command')
        #カメラコマンド
        self.send_command('streamon')
        self.send_command('takeoff')

        # ビデオに送る回線コマンド
        self.cap =None
        self.video_addr='udp://@'+self.host_ip+':'+str(self.video_port)

        self.face_cascade =cv2.CascadeClassifier(FACE_DETECT_XML_FILE)
        self.eye_cascade =cv2.CascadeClassifier(EYE_DETECT_XML_FILE)

        self.video(self.video_addr,self.cap,self.face_cascade, self.eye_cascade)


    def send_command(self,command):
        # データ送信
        self.socket.sendto(command.encode('utf-8'),self.drone_address)

    def receive_data(self):
        while True:
            try:
                self.response,_ =self.socket.recvfrom(1024)
            except Exception as e:
                print('エラーじゃ！')
                print(e)
                # break

    def video(self,addr,cap,face,eye):
        if cap is None:
            cap =cv2.VideoCapture(addr)
        elif not cap.isOpend():
            cap.open(addr)

        while True:
            ret,frame =cap.read()
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces =face.detectMultiScale(grey,1.3,5)
            print(len(faces))
            # facesの画像の座標の高さを取得できる
            # wとhは幅と高さ
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                eye_gray = grey[y:y + h, x:x + w]
                eye_color = frame[y:y + h, x:x + w]
                eyes = eye.detectMultiScale(eye_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(eye_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    def takeoff(self):
        return self.send_command('takeoff')

    def land(self):
        return  self.send_command('land')

    def flip_back(self):
        return  self.send_command('flip b')




if __name__ == '__main__':
    tello =Tello()
    tello.takeoff()
    time.sleep(5)
    tello.flip_back()
    tello.land()