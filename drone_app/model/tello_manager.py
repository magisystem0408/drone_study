import socket
import threading
import time



class Tello:
    def __init__(self,host_ip: object = '192.168.10.2', host_port: object = 8889
                 ,drone_ip:object='192.168.10.1',drone_port:object=8889):

        # こちらの情報
        self.host_ip =host_ip
        self.host_port=host_port
        #socket
        self.socket =socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.video =socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

        self.socket.bind((self.host_ip,host_port))

        #ドローンの情報
        self.drone_ip = drone_ip
        self.drone_port =drone_port
        #socket
        self.drone_address =(self.drone_ip,self.drone_port)

        self.send_command('command')

        self.video_port=11111

        self.response =None
        self.stop_event=threading.Event()


    def send_command(self,command):
        # データ送信
        self.socket.sendto(command.encode('utf-8'),self.drone_address)


    # def recive_response(self,stop_event):

    def takeoff(self):
        return self.send_command('takeoff')

    def land(self):
        return  self.send_command('land')

    def flip_back(self):
        return  self.send_command('flip b')


    def _receive_video_thred(self):
        packet_data =""
        # while True:


    def _h264_decode(self,packet_data):
        res_frame_list =[]



if __name__ == '__main__':
    tello =Tello()
    tello.takeoff()
    time.sleep(5)
    tello.flip_back()
    tello.land()