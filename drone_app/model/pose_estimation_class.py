import numpy as np

"""
ポーズ判定など、3次元姿勢予測によって得られた座標から
計算とモデルを判定するクラス
"""
class PoseDetector:

    def __init__(self, left_arm_angle, right_arm_angle, left_body_angle, right_body_angle):
        self.left_arm_angle = left_arm_angle
        self.right_arm_angle = right_arm_angle
        self.left_body_angle = left_body_angle
        self.right_body_angle = right_body_angle


    @staticmethod
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


    def pose_T(self):
        output = False
        if self.left_arm_angle >= 150 and self.right_arm_angle >= 150 and (
                self.right_body_angle <= 120 and self.right_body_angle >= 90) and(self.left_body_angle <=120 and self.right_body_angle >=90 ):
            print("マムシ")
            return True
        return False

    def pose_A(self):
        output = False
        if self.left_body_angle >= 120 and self.right_body_angle >= 120:
            return True
        return False

    def pose_L(self):
        output = False
        if (self.left_body_angle <= 120 and self.left_body_angle >= 90) and self.right_body_angle >= 150:
            return True

        if (self.right_body_angle <= 120 and self.right_body_angle >= 90) and self.left_body_angle >= 150:
            return True
        return False
