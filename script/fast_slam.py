#!/usr/bin/env python

#   roslaunch husky_gazebo husky_playpen.launch
#   rosrun key_teleop key_teleop.py key_vel:=/husky_velocity_controller/cmd_vel


import time
import cv2

import rospy
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import LaserScan
import tf 

import numpy as np
from numpy import pi, abs
import numpy.linalg as la
import math
from math import cos, sin, sqrt, atan2, exp
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

class GridMap:

    def __init__(self, height=211, width=211, accuracy=0.1):
        if height % 2 == 0:
            height += 1
        if width % 2 == 0:
            width += 1
        self.height = height
        self.width = width
        self.acc = accuracy
        self.pixelOverMeter = 1/accuracy
        self.map = np.zeros((height, width, 3), dtype=np.uint8) 
    
    def metersToPixelsIndex(self, pose):
        row = int(np.round(self.height/2 - self.pixelOverMeter * pose[1]))
        col = int(np.round(self.pixelOverMeter * pose[0] + self.width/2))
        return (row, col)

    def PixelToMeter(self, i, j):
        y = (self.height/2 - i) * self.acc 
        x = (j - self.width/2 ) * self.acc
        return (x, y)
    
    def ray_cast(self, xco, yco, Zmin=1.0, Zmax=30.0, debug=True):
        # xco is coordinate of the horiontal axis which points right and has w as max value.
        # yco is coordinate of the virtical axis which points up and has h as max value.
        pixelToMeterRatio = self.acc
        image = self.map
        if debug:
            h, w, c = image.shape
            assert c == 3, "image must be an RGB image"
        else:
            h, w = image.shape
        xc = h-1 - yco
        yc = xco

        r = np.linspace(Zmin/pixelToMeterRatio, Zmax/pixelToMeterRatio, 1000)
        n = 8
        thetastep = 2*pi/n
        ranges = np.zeros((n,))
        angles = np.zeros((n,))
        for i in range(n):
            theta = -thetastep * i
            angles[i] = theta
            x = xc + r*sin(theta)
            y = yc + r*cos(theta)
        
            xint = np.int32(np.round(x))
            yint = np.int32(np.round(y))

            index = np.logical_and(np.logical_and(xint < w, xint >= 0), np.logical_and(yint < h, yint >= 0))
            xint = xint[index]
            yint = yint[index]

            if debug:
                new_set = image[xint, yint, 0]
            else:
                new_set = image[xint, yint]

            if new_set.size:
                intersection_index = np.argmax(new_set)
                px, py = xint[intersection_index], yint[intersection_index]
                if debug:
                    image[xint, yint, 1] = 255
                    if image[px, py, 0] == 0:
                        ranges[i] = Zmax
                        continue
                else:
                    if image[px, py] == 0:
                        ranges[i] = Zmax
                        continue                
                ranges[i] = sqrt( (px - xc)**2 + (py - yc)**2  )*pixelToMeterRatio
                if debug: #plot the intersection as a point
                    cv2.circle(image, (py, px), radius=2, color=(0, 0, 255), thickness=-1)
            else: # the ray did not hit any obstical in the image 
                ranges[i] = Zmax
        if debug:
            print(angles)
            print(ranges)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

class Robot:

    def __init__(self, N, alphas = [0.3, 0.08, 0.08, 0.3], alpha=0.3, beta=0.0066, Phit_segma=1.0, lambda_short=0.3):
        self.N = N
        self.alphas = alphas
        self.firstOdometry = True
        self.firstLinkStates = True
        self.firstLaserScan = True
        self.sensorAlpha = alpha
        self.sensorBeta = beta
        self.gmap = GridMap()

        self.Zmin = 0
        self.Phit_semga = Phit_segma
        self.lambda_short = lambda_short


    def calcProbabilities(self, Ztk, Zstar):
        #calc_Phit and calc_Prand
        if Ztk <= self.Zmin or Ztk => self.Zmax:
            Phit = 0
            Prand = 0
        else:
            Phit = (1/sqrt(2*pi*self.Phit_segma^2)) * exp(-0.5*(Ztk-Zstar)^2/Phit_segma^2)
            Prand = 1.0/(self.Zmax-self.Zmin)
        #calc_Pmax
        Pmax = int(Ztk == self.Zmax)
        #calc_Pshort
        if Ztk < self.Zmin or Ztk > self.Zstar:
            Pshort = 0
        else:
            n = 1/(1-exp(-self.lambda_short * self.Zstar))
            Pshort = n * self.lambda_short * exp(-self.lambda_short*Ztk)

        return [Phit, Pshort, Pmax, Prand]
       
    def odometryCallback(self, msg):
        self.currOdom = self.getOdomPose(msg)
        if self.firstOdometry:
            self.firstOdometry = False
            self.prevOdom = self.currOdom
        self.currPoses = self.sample_motion_model()
        self.prevOdom = self.currOdom
        self.prevPoses = self.currPoses

    def linkStatesCallback(self, msg):
        # print('hello from ROBOT linkstates callback ')

        if self.firstLinkStates:
            self.firstLinkStates = False
            names = np.array(msg.name)
            self.robotIndex = np.argmax(names == '/::base_link')
            self.worldPose = self.getWorldPose(msg, self.robotIndex)
            self.prevOdom = self.worldPose
            self.prevPoses = np.zeros((self.N, 3))
            self.prevPoses[:] = self.worldPose
        self.worldPose = self.getWorldPose(msg, self.robotIndex)

    def laserScanCallback(self, msg):
        if self.firstLaserScan:
            self.firstLaserScan = False
            self.scanAgles = np.linspace(msg.angle_max, msg.angle_min, len(msg.ranges))
            self.Zmax = msg.range_max
        self.Zt = msg.ranges        

    def getOdomPose(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return np.array([x, y, euler[2]])

    def getWorldPose(self, msg, robotIndex):
        x = msg.pose[robotIndex].position.x
        y = msg.pose[robotIndex].position.y
        quaternion = (
            msg.pose[robotIndex].orientation.x,
            msg.pose[robotIndex].orientation.y,
            msg.pose[robotIndex].orientation.z,
            msg.pose[robotIndex].orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        return np.array([x, y, euler[2]])

    def sample_motion_model(self):
        drot1 = math.atan2(self.currOdom[1] - self.prevOdom[1], self.currOdom[0] - self.prevOdom[0]) - self.prevOdom[2]
        dtrans = math.sqrt((self.currOdom[0] - self.prevOdom[0])**2 + (self.currOdom[1] - self.prevOdom[1])**2)
        drot2 = self.currOdom[2] - self.prevOdom[2] - drot1
        currPoses = np.zeros_like(self.prevPoses)
        for i, prevPose in enumerate(self.prevPoses):
            drot1_hat = drot1 - self.sample(self.alphas[0]*drot1 + self.alphas[1]*dtrans)
            dtrans_hat = dtrans - self.sample(self.alphas[2]*dtrans + self.alphas[3]*(drot1 + drot2))
            drot2_hat = drot2 - self.sample(self.alphas[0]*drot2 + self.alphas[1]*dtrans)
            changeInPose = np.array([dtrans_hat*cos(prevPose[2] + drot1_hat),\
                dtrans_hat*sin(prevPose[2] + drot1_hat),\
                drot1_hat + drot2_hat])
            currPoses[i] = prevPose + changeInPose
        return currPoses

    def sample(self, b):
        rn = 2*np.random.rand(12,1) - 1 
        return b*rn.sum()/6

    def inverse_range_sensor_model(self, xi, yi):
        r = sqrt((xi-self.worldPose[0])**2 + (yi-self.worldPose[1])**2)
        phai = atan2(yi-self.worldPose[1], xi-self.worldPose[0]) - self.worldPose[2]
        phai = phai%(2*pi) if phai >= 0 else -(-phai%(2*pi))
        if phai > pi:
            phai = phai - 2*pi
        elif phai < (-pi):
            phai = phai + 2*pi
        diff = np.abs(phai-self.scanAgles)
        k = np.argmin(diff) # this can be faster
        if phai > np.pi or phai < -np.pi:
            print('phai {} is too big'.format(phai))
            print(atan2(yi-self.worldPose[1], xi-self.worldPose[0]), self.worldPose[2])
        # print(phai, self.worldPose[2], diff[k], k)
        if (r > min(self.Zmax, self.Zt[k] + self.sensorAlpha/2 )) or (diff[k] > self.sensorBeta/2) :
            value = 0
        elif (self.Zt[k] < self.Zmax) and (np.abs(r-self.Zt[k]) < self.sensorAlpha/2):
            value = 60
        elif r <= self.Zt[k]:
            value = -30
        else:
            value = 0
            print("Error in inverse_range_sensor_model with phai={} and robotWangle={}, xi={},yi={}"\
                .format(phai, self.worldPose[2], xi, yi))        
        return value

    def buildMap(self):
        self.gmap.map[:, :, 1] = 0
        self.gmap.map[:, :, 2] = 0
        for i in range(self.gmap.height):
            for j in range(self.gmap.width):
                xi, yi = self.gmap.PixelToMeter(i, j)
                value = self.inverse_range_sensor_model(xi, yi)
                #tmp = self.gmap.map[i, j, 0] + value
                #self.gmap.map[i, j, 0] = max(0, min(tmp, 255))
                self.gmap.map[i, j, 0] = np.clip(self.gmap.map[i, j, 0] + value, 0, 255)
                # if value > 0:
                #     self.gmap.map[i, j, 1] = 255

        i, j = self.gmap.metersToPixelsIndex(self.worldPose)
        xi, yi = self.gmap.PixelToMeter(i, j)
        cv2.circle(self.gmap.map, (j,i), radius=2, color=(0, 0, 255), thickness=-1)
        
        rotated_angles = self.scanAgles + self.worldPose[2]
        for i, angle in enumerate(rotated_angles):
            if self.Zt[i] == float('inf'):
                continue
            x = self.worldPose[0] + self.Zt[i] *np.cos(angle)
            y = self.worldPose[1] + self.Zt[i] *np.sin(angle)
            i, j = self.gmap.metersToPixelsIndex([x, y])
            if(i>210 or j>210):
                continue
            self.gmap.map[i, j, 2] = 255 
            x = self.worldPose[0] + 2 *np.cos(angle)
            y = self.worldPose[1] + 2 *np.sin(angle)
            i, j = self.gmap.metersToPixelsIndex([x, y])
            if(i>210 or j>210):
                continue
            self.gmap.map[i, j, 1] = 255 



def main():

    rospy.init_node('fast_slam', anonymous=True)
    robot = Robot(40)
    rospy.Subscriber('/gazebo/link_states', LinkStates, robot.linkStatesCallback)
    time.sleep(0.01)
    rospy.Subscriber('/husky_velocity_controller/odom', Odometry, robot.odometryCallback)
    rospy.Subscriber('/scan', LaserScan, robot.laserScanCallback)


    fig = plt.figure()
    time.sleep(0.01)
    while not rospy.is_shutdown():
        # print(abs(xi - robot.worldPose[0]), abs(yi - robot.worldPose[1]) )
        robot.buildMap()
        cv2.imshow('gmap',robot.gmap.map)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite('map.jpg', robot.gmap.map[:, :, 0])





if __name__ == '__main__':

    try:
        main()
        # gmap = GridMap()
        # i, j = gmap.metersToPixelsIndex([5, -3])
        # print(i, j)
        # gmap.map[i, j] = 255
        # gmap.map = cv2.circle(gmap.map, (100,100), radius=2, color=(0, 0, 255), thickness=-1)

        # cv2.imshow('image', gmap.map) 
        # cv2.waitKey(0)
    except rospy.ROSInterruptException:
        pass
