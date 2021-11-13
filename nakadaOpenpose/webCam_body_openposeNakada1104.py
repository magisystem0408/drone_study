# From Python
# It requires OpenCV installed for Python
import os, sys
from sys import platform
import argparse

# サードパーティー
import cv2
import numpy as np

import time
import datetime
import math #20200814
from math import pi, atan2
from math import atan2, degrees, sqrt
from simple_pid import PID
import pyautogui

# 作成した関数
from OP import *
from models.FPS import FPS

#グローバル変数群
op = OP(number_people_max=10, min_size=5)
detected = False
body_size = None
shoulders_width =None

pose = None

def distance (A, B):
    """
        Calculate the square of the distance between points A and B
    """
    return int(sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2))

def angle (A, B, C):
    """
        Calculate the angle between segment(A,p2) and segment (p2,p3)
    """
    if A is None or B is None or C is None:
        return None
    return degrees(atan2(C[1]-B[1],C[0]-B[0]) - atan2(A[1]-B[1],A[0]-B[0]))%360

def vertical_angle (A, B):
    """
        Calculate the angle between segment(A,B) and vertical axe
    """
    if A is None or B is None:
        return None
    return degrees(atan2(B[1]-A[1],B[0]-A[0]) - pi/2)

def quat_to_yaw_deg(qx,qy,qz,qw):
    """
        Calculate yaw from quaternion
    """
    degree = pi/180
    sqy = qy*qy
    sqz = qz*qz
    siny = 2 * (qw*qz+qx*qy)
    cosy = 1 - 2*(qy*qy+qz*qz)
    yaw = int(atan2(siny,cosy)/degree)
    return yaw


body_kp_id_to_name = {
    0: "Nose",
    1: "Neck",
    2: "RShoulder",
    3: "RElbow",
    4: "RWrist",
    5: "LShoulder",
    6: "LElbow",
    7: "LWrist",
    8: "MidHip",
    9: "RHip",
    10: "RKnee",
    11: "RAnkle",
    12: "LHip",
    13: "LKnee",
    14: "LAnkle",
    15: "REye",
    16: "LEye",
    17: "REar",
    18: "LEar",
    19: "LBigToe",
    20: "LSmallToe",
    21: "LHeel",
    22: "RBigToe",
    23: "RSmallToe",
    24: "RHeel",
    25: "Background"}




def check_target(frame, w, h, nbody): #20200814

#We chose the person wearing red patch
#and detect the color not BGR but HSV 

    TARGET_H = 2*180.0/180*math.pi # 175 Key Point
    DELTA_H = 2*4.0/180*math.pi #7.5 10 is maybe OK
    MAX_S = 255
    MIN_S = 180 #40
    MIN_V = 100 #15
    temp_i = -1 #20201202 add code Nakada
    min_distance2 = h**2 + w**2 #20201202 add code Nakada
    print(min_distance2)
    min_neck = None #20201202 add code Nakada


    for i in range(nbody):
            r_elbow = op.get_body_kp("RElbow", i)
            l_elbow = op.get_body_kp("LElbow", i)
            r_shoulder = op.get_body_kp("RShoulder", i)
            l_shoulder = op.get_body_kp("LShoulder", i)
            neck = op.get_body_kp("Neck") #20201202 add code Nakada

            if r_elbow:
                if r_shoulder: 
                    for j in range(1,5):
                        alpha = j/6.0
                        x1 = int((alpha * r_elbow[0] + (1-alpha) * r_shoulder[0])+0.5)
                        y1 = int((alpha * r_elbow[1] + (1-alpha) * r_shoulder[1])+0.5)
                
                        if x1 < 0:
                            x1 = 0
                        if x1 >= w:
                            x1 = w-1
                        if y1 < 0:
                            y1 = 0
                        if y1 >= h:
                            y1 = h-1

                        print("larget right position and col1:")
                        print(x1)
                        print(y1)
                        #print(w)
                        #print(h)
                        col1 = frame[y1, x1]#Difine frame[y1, x1]'s BGR as col1
                        H = 2.0*col1[0]/180.0*math.pi
                        S = col1[1]
                        V = col1[2]
                        #print(col1)
                        #col1_1 = HSV_frame[y1, x1] #20200814
                        #print(col1_1)
                
                        #if col1[0] >= 160 and col1[1] >= 160 and col1[2] >= 160:
                        #    return i

                        if 1.0-math.cos(H-TARGET_H) <= 1.0 - math.cos(DELTA_H) and S >= MIN_S and S <= MAX_S and V >= MIN_V:
                            detected = True #20201202 add code Nakada
                            #self.last_detected_time = 0 #20201202 add code Nakada
                            print(col1)
                            #if neck:    
                            #    self.last_detected_pos = neck #20201202 add code Nakada
                            #    print("neck:")
                            #    print(neck)
                            #else:
                            #    self.last_detected_pos = None #20201202 add code Nakada
                            return i

            if l_elbow:
                if l_shoulder: 
                    for j in range(1,5):
                        alpha = j/6.0
                        x2 = int((alpha * l_elbow[0] + (1-alpha) * l_shoulder[0])+0.5)
                        y2 = int((alpha * l_elbow[1] + (1-alpha) * l_shoulder[1])+0.5)
                
                        if x2 < 0:
                            x2 = 0
                        if x2 >= w:
                            x2 = w-1
                        if y2 < 0:
                            y2 = 0
                        if y2 >= h:
                            y2 = h-1

                        print("larget left position and col2:")
                        print(x2)
                        print(y2)
                        #print(w)
                        #print(h)
                        col2 = frame[y2, x2]
                        H = 2.0*col2[0]/180.0*math.pi
                        S = col2[1]
                        V = col2[2]
                        #print(col2)
                        #col1_1 = HSV_frame[y1, x1] #20200814
                        #print(col1_1)
                
                        #if col1[0] >= 160 and col1[1] >= 160 and col1[2] >= 160:
                        #    return i

                        if 1-math.cos(H-TARGET_H) <= 1 - math.cos(DELTA_H) and S >= MIN_S and S <= MAX_S and V >= MIN_V:
                            detected = True #20201202 add code Nakada
                            #self.last_detected_time = 0 #20201202 add code Nakada
                            print(col2)
                            #if neck:    
                            #    self.last_detected_pos = neck #20201202 add code Nakada
                            #    print("neck:")
                            #    print(neck)
                            #else:
                            #    self.last_detected_pos = None #20201202 add code Nakada
                            return i

            #if neck and self.last_detected_pos and not self.detected: #20201202 add code Nakada
            #    neck_distance2 = (neck[0] - self.last_detected_pos[0])**2 + (neck[1] - self.last_detected_pos[1])**2
            #    print(self.last_detected_pos)
            #    print(neck)
            #    print("distance:")
            #    print(i)
            #    print(neck_distance2)
            #    if neck_distance2 <= min_distance2 and neck_distance2 <= 100:
            #        temp_i = i
            #        min_distance2 = neck_distance2
            #        min_neck = neck

    detected = False #20201202 add code Nakada
    #self.last_detected_time = self.last_detected_time + 1 #20201202 add code Nakada

    #if nbody > 0 and temp_i >-1 and min_neck: #20201202 add code Nakada
    #    if self.last_detected_time <= 10:
    #        self.last_detected_pos = min_neck
    #        return temp_i
     
    return -1

def check_pose2(w, h, pid):
    """
        Check if we detect a pose in the body detected by Openpose
    """

    neck = op.get_body_kp("Neck", pid)
    r_wrist = op.get_body_kp("RWrist", pid)
    l_wrist = op.get_body_kp("LWrist", pid)
    r_elbow = op.get_body_kp("RElbow", pid)
    l_elbow = op.get_body_kp("LElbow", pid)
    r_shoulder = op.get_body_kp("RShoulder", pid)
    l_shoulder = op.get_body_kp("LShoulder", pid)
    r_ear = op.get_body_kp("REar", pid)
    l_ear = op.get_body_kp("LEar", pid) 
    
    shoulders_width = distance(r_shoulder,l_shoulder) if r_shoulder and l_shoulder else None
    midhip = op.get_body_kp("MidHip", pid)
    nose = op.get_body_kp("Nose", pid)
    if neck and midhip:
         body_size = distance(neck, midhip) 
    elif neck and nose:
         body_size = 2.3 * distance(neck, nose) 
    elif shoulders_width:
         body_size = 1.5 * self.shoulders_width

    vert_angle_right_arm = vertical_angle(r_wrist, r_elbow)
    vert_angle_left_arm = vertical_angle(l_wrist, l_elbow)

    left_hand_up = neck and l_wrist and l_wrist[1] < neck[1]
    right_hand_up = neck and r_wrist and r_wrist[1] < neck[1]

    if right_hand_up:
        if left_hand_up:
            #Both hands up
            if r_ear and (r_ear[0]-neck[0])*(r_wrist[0]-neck[0])>0 and l_ear and (l_ear[0]-neck[0])*(l_wrist[0]-neck[0])>0:
            #Right ear and right hand on the same side and left ear and left hand on the same side
                if vert_angle_right_arm and vert_angle_left_arm:
                    #front arms up open
                    if -70 < vert_angle_right_arm < -40 and 40 < vert_angle_left_arm < 70: #2020901 add code
                        return "BOTH_ARMS_UP_OPEN_FRONT" #20200131 add code
                    #back arms up open
                    elif -70 < vert_angle_left_arm < -40 and 40 < vert_angle_right_arm < 70: #2020901 add code
                        return "BOTH_ARMS_UP_OPEN_BACK" #20200131 add code
                    #front check mark pose
                    elif -15 < vert_angle_right_arm < 15 and 75 < vert_angle_left_arm < 105: #20200902 add code change 20201118 uega 0
                        return "CHECK_MARK_POSE_FRONT" #20200902 add code
                    #back check mark pose
                    elif -15 < vert_angle_left_arm < 15 and 75 < vert_angle_right_arm < 105: #20200902 add code change 20201118 uega 0
                        return "CHECK_MARK_POSE_BACK" #20201209 add code
                    #front reverse check mark pose
                    elif -105 < vert_angle_right_arm < -75 and -15 < vert_angle_left_arm < 15: #20200902 add code change 20201118
                        return "REVERSE_CHECK_MARK_POSE_FRONT" #20200902 add code
                        #self.drone.takeoff() #20200211 add code
                    #back reverse check mark pose
                    elif -105 < vert_angle_left_arm < -75 and -15 < vert_angle_right_arm < 15: #20201209 add code
                        return "REVERSE_CHECK_MARK_POSE_BACK" #20201209 add code
                    #front arms up close
                    elif -20 <= vert_angle_right_arm <= 10 and -10 < vert_angle_left_arm < 20:#2020901 add code
                        return "BOTH_ARMS_UP_CLOSE_FRONT" #20200131 add code
                    #back arms up close
                    elif -20 <= vert_angle_left_arm <= 10 and -10 < vert_angle_right_arm < 20:#2020901 add code
                        return "BOTH_ARMS_UP_CLOSE_BACK" #20200131 add code
    """
    if right_hand_up:
        if not left_hand_up:
            # Only right arm up
            if r_ear and (r_ear[0]-neck[0])*(r_wrist[0]-neck[0])>0:
            # Right ear and right hand on the same side
                return None
                if vert_angle_right_arm:
                    if vert_angle_right_arm < -15:
                        return "RIGHT_ARM_UP_OPEN"
                    if 15 < vert_angle_right_arm < 90:
                        return "RIGHT_ARM_UP_CLOSED"
            elif l_ear and self.shoulders_width and distance(r_wrist,l_ear) < self.shoulders_width/4:
                # Right hand close to left ear
                return "RIGHT_HAND_ON_LEFT_EAR"
        else:
            # Both hands up
            # Check if both hands are on the ears
            if r_ear and l_ear:
                ear_dist = distance(r_ear,l_ear)
                if distance(r_wrist,r_ear)<ear_dist/3 and distance(l_wrist,l_ear)<ear_dist/3:
                    return("HANDS_ON_EARS")
            # Check if boths hands are closed to each other and above ears 
            # (check right hand is above right ear is enough since hands are closed to each other)
            if self.shoulders_width and r_ear:
                near_dist = self.shoulders_width/3
                #if r_ear[1] > r_wrist[1] and distance(r_wrist, l_wrist) < near_dist :
                if r_ear[0] > r_wrist[0] and distance(r_wrist, l_wrist) < near_dist :
                    return "CLOSE_HANDS_UP"

    else:
        if left_hand_up:
            # Only left arm up
            if l_ear and (l_ear[0]-neck[0])*(l_wrist[0]-neck[0])>0:
                return None
                # Left ear and left hand on the same side
                if vert_angle_left_arm:
                    if vert_angle_left_arm < -15:
                        return "LEFT_ARM_UP_CLOSED"
                    if 15 < vert_angle_left_arm < 90:
                        return "LEFT_ARM_UP_OPEN"
            elif r_ear and self.shoulders_width and distance(l_wrist,r_ear) < self.shoulders_width/4:
                # Left hand close to right ear
                return "LEFT_HAND_ON_RIGHT_EAR"
        else:
            # Both wrists under the neck
            if neck and self.shoulders_width and r_wrist and l_wrist:
                near_dist = self.shoulders_width/3
                if distance(r_wrist, neck) < near_dist and distance(l_wrist, neck) < near_dist :
                    return "HANDS_ON_NECK"
    return None
    """

cap = cv2.VideoCapture(1)

while True:
    ret, raw_frame = cap.read()
    if not ret:
        break

    frame = raw_frame.copy()
    frame2 = frame
    
    

    h,w,_ = frame2.shape
    nb_people, pose_kps, face_kps = op.eval(frame2)
    print("nb:" + str(nb_people))
    
    
    if nb_people > 0:
        HSV_frame = cv2.cvtColor(np.array(frame2,dtype=np.uint8), cv2.COLOR_BGR2HSV)
        op.draw_body(frame2)

        pid = check_target(HSV_frame, w, h, nb_people)
        print("pid***:")
        print(pid)
        

        
        if pid >= 0:
            pose = check_pose2(w,h, pid)
            print("pose")
            print(pose)
            print("shoulder width")
            print(shoulders_width)
        
            if pose:
                # We trigger the associated action
                #log.info(f"pose detected : {self.pose}")
                if pose == "BOTH_ARMS_UP_OPEN_FRONT" or pose == "BOATH_ARMS_UP_OPEN_BACK": #20201209 update
                    #log.info("TAKE OFF from pose") #20200901 add code
                    #log.info("GOING UP from pose") #20200901 add code
                    #self.axis_speed["throttle"] = self.def_speed["throttle"] #20200131 add code
                    #self.set_speed("throttle", self.def_speed["throttle"])
                    #self.drone.takeoff() #20200901 add code
                    pyautogui.press('f')
                    
                    shoulders_width = None #20200903 nakada
                    body_size = None #20200903 nakada
                    ##self.tracking = True   #20201021 add code
                    #l_diff = 0  #20201021 add code
                    #r_diff = 0  #20201021 add code
                    #pid_pitch = PID(0.7,0.04,0.3,setpoint=0,output_limits=(-50,50)) #20200903 nakada 0.5  0.6
                    #self.pid_roll = PID(1.8,0.9,0.3,setpoint=0,output_limits=(-50,50)) #20201021 ogawa
                    #pid_roll = PID(1.4, 1.0, 0.1,setpoint=0,output_limits=(-40,40)) #20201118 ogawa
                    #pid_yaw = PID(0.20, 0,0,setpoint=0,output_limits=(-100, 100)) #20201021
                    #pid_throttle = PID(0.40,0,0,setpoint=0,output_limits=(-80,100)) #2020908 0.4 to 0.35
                elif pose == "BOTH_ARMS_UP_CLOSE_FRONT" or pose == "BOTH_ARMS_UP_CLOSE_BACK": #20200901 add code : #20201209 update
                    #log.info("RAND from pose") 
                    #self.tracking = False 
                    #self.toggle_tracking(tracking=False)
                    #self.keep_distance = None 
                    #self.drone.land() #20200901 add code
                    pyautogui.press('b')        
        
            max_x = 0
            max_y = 0
            min_x = w-1
            min_y = h-1
            rect_flg = False

            for i in range(24):
                xy = op.get_body_kp(body_kp_id_to_name[i], pid)
                if xy:
                    rect_flg = True
                    if xy[0] < min_x:
                        min_x = xy[0]
                    if xy[1] < min_y:
                        min_y = xy[1]
                    if xy[0] > max_x:
                        max_x = xy[0]
                    if xy[1] > max_y:
                       max_y = xy[1]
                    if max_y >= h:
                       max_y = h-1
                    if min_y < 0:
                       min_y = 0
                    if max_x >= w:
                       max_x = w-1
                    if min_x < 0:
                       min_x = 0

            if rect_flg:
                if detected:
                    cv2.rectangle(frame2, (min_x, min_y), (max_x, max_y), (0, 0, 255), thickness=10)
                    print("draw rect!")
                else:
                    cv2.rectangle(frame2, (min_x, min_y), (max_x, max_y), (0, 255, 255), thickness=10) #20201202 add code Nakada
                    print("draw rect!")
                            
    cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", frame2)

    #cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", frame)
    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))

   # print("Right hand keypoints: \n" + str(datum.eyesKeypoints))


    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey(0)


    # cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destoryAillWindows()
