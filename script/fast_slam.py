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


def ray_cast(xco, yco, image, scanAngles, Zmin=1.0, Zmax=30.0, pixelToMeterRatio=0.1, debug=False):
    if debug:
        h, w, c = image.shape
        assert c == 3, "image must be an RGB image"
    else:
        h, w = image.shape
    xc = xco
    yc = yco
    r = np.linspace(Zmin/pixelToMeterRatio, Zmax/pixelToMeterRatio, 1000)
    n = scanAngles.shape[0]
    ranges = np.zeros((n,))
    for i in range(n):
        theta = -scanAngles[i]
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
            if intersection_index == 0:
                print('the intersection_index is equal to zero!')
            px, py = xint[intersection_index], yint[intersection_index]
            if debug:
                #image[xint[0:50], yint[0:50], 1] = 255
                if image[px, py, 0] == 0:
                    ranges[i] = Zmax
                    continue
            else:
                if image[px, py] == 0:
                    ranges[i] = Zmax
                    continue              
            ranges[i] = sqrt( (px - xc)**2 + (py - yc)**2  )*pixelToMeterRatio
            if debug: #plot the intersection as a point
                pass #image[px-1:px+1, py-1:py+1, 2] = 255
        else: # the ray did not hit any obstical in the image 
            ranges[i] = Zmax
    if debug:
        pass # print(ranges)

    return ranges



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

    def __init__(self, N, scanNum=20, alphas=[0.05, 0.05, 0.05, 0.05], alpha=0.3, beta=0.0066, Phit_segma=0.6,\
                                    lambda_short=0.3, prob_weights=[0.95, 0.01, 0.02, 0.02], movement_thresh=0.1):
        self.N = N
        self.alphas = alphas
        self.firstOdometry = True
        self.firstLinkStates = True
        self.firstLaserScan = True
        self.sensorAlpha = alpha
        self.sensorBeta = beta
        self.gmaps = np.array([GridMap() for _ in range(N)])
        self.scanNum = scanNum
        self.Zmin = 0
        self.Phit_segma = Phit_segma
        self.lambda_short = lambda_short
        self.prob_weights = np.array(prob_weights, np.float64)
        self.movement_thresh = movement_thresh
        self.weights = np.ones((N,))* 1/N

    def calcProbabilities(self, Ztk, Zstar):
        #calc_Phit and calc_Prand
        if Ztk < self.Zmin or Ztk > self.Zmax:
            Phit = 0
            Prand = 0
        else:
            Phit = (1/sqrt(2*pi*self.Phit_segma**2)) * exp(-0.5*(Ztk-Zstar)**2/(self.Phit_segma**2))
            Prand = 1.0/(self.Zmax-self.Zmin)
        #calc_Pmax
        Pmax = int(Ztk == self.Zmax)
        #calc_Pshort
        if Ztk < self.Zmin or Ztk > Zstar:
            Pshort = 0
        else:
            n = 1/(1-exp(-self.lambda_short * Zstar))
            Pshort = n * self.lambda_short * exp(-self.lambda_short*Ztk)
        probs = np.array([Phit, Pshort, Pmax, Prand], dtype=np.float64) 
        #print('{} -> {} => {}'.format(Ztk, Ztk-Zstar, probs))
        return probs

    def odometryCallback(self, msg):
        newOdom = self.getOdomPose(msg)
        if self.firstOdometry:
            self.firstOdometry = False
            self.currOdom = newOdom
            self.lastOdomLocalization = self.currOdom
            return
        self.currPoses = self.sample_motion_model(newOdom)
        self.prevOdom = self.currOdom
        self.currOdom = newOdom

    def linkStatesCallback(self, msg):
        if self.firstLinkStates:
            self.firstLinkStates = False
            names = np.array(msg.name)
            self.robotIndex = np.argmax(names == '/::base_link')
            self.worldPose = self.getWorldPose(msg, self.robotIndex)
            self.currOdom = self.worldPose
            self.currPoses = np.zeros((self.N, 3))
            self.currPoses[:] = self.worldPose
        self.worldPose = self.getWorldPose(msg, self.robotIndex)
    def laserScanCallback(self, msg):
        if self.firstLaserScan:
            self.firstLaserScan = False
            self.scanAgles = np.linspace(msg.angle_max, msg.angle_min, len(msg.ranges))
            self.angleMin = msg.angle_min
            self.angleMax = msg.angle_max
            self.Zmax = msg.range_max
            ratio = self.scanAgles.shape[0]/self.scanNum
            self.K_scanAngles_indices = np.array(range(ratio/2, self.scanAgles.shape[0] + ratio/2, ratio))
            self.K_scanAngles = self.scanAgles[self.K_scanAngles_indices]
        ranges = np.array(msg.ranges)
        ranges[ranges[:]==float('inf')] = self.Zmax
        self.Zt = ranges #np.array(msg.ranges) 

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
    def sample_motion_model(self, newOdom):
        drot1 = math.atan2(newOdom[1] - self.currOdom[1], newOdom[0] - self.currOdom[0]) - self.currOdom[2]
        dtrans = math.sqrt((newOdom[0] - self.currOdom[0])**2 + (newOdom[1] - self.currOdom[1])**2)
        drot2 = newOdom[2] - self.currOdom[2] - drot1
        newPoses = np.zeros_like(self.currPoses)
        for i, prevPose in enumerate(self.currPoses):
            drot1_hat = drot1 - self.sample(self.alphas[0]*drot1 + self.alphas[1]*dtrans)
            dtrans_hat = dtrans - self.sample(self.alphas[2]*dtrans + self.alphas[3]*(drot1 + drot2))
            drot2_hat = drot2 - self.sample(self.alphas[0]*drot2 + self.alphas[1]*dtrans)
            changeInPose = np.array([dtrans_hat*cos(prevPose[2] + drot1_hat),\
                dtrans_hat*sin(prevPose[2] + drot1_hat),\
                drot1_hat + drot2_hat])
            newPoses[i] = prevPose + changeInPose
        return newPoses
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

    def plotZt(self, image, pose):
        image[:, :, 1] = 0
        image[:, :, 2] = 0
        # rotated_angles = self.K_scanAngles + self.worldPose[2]
        for i in self.K_scanAngles_indices:
            angle = self.scanAgles[i] + pose[2]
            if self.Zt[i] == float('inf'):
                continue
            x = pose[0] + self.Zt[i] *np.cos(angle)
            y = pose[1] + self.Zt[i] *np.sin(angle)
            i, j = self.gmaps[0].metersToPixelsIndex([x, y])
            if(i>210 or j>210):
                continue
            image[i-1:i+1, j-1:j+1, 1] = 255 
            x = pose[0] + 2 *np.cos(angle)
            y = pose[1] + 2 *np.sin(angle)
            i, j = self.gmaps[0].metersToPixelsIndex([x, y])
            if(i>210 or j>210):
                continue
            image[i-1:i+1, j-1:j+1, 1] = 255
            #for pose in self.currPoses:
            i, j = self.gmaps[0].metersToPixelsIndex(pose)
            image[i-1:i+1, j, 2] = 255
            image[i, j-1:j+1, 2] = 255  

            # i, j = self.gmaps[0].metersToPixelsIndex(pose)
            # image[i-1:i+1, j, 1] = 255
            # image[i, j-1:j+1, 1] = 255

    def update_occupancy_grid(self, xt, prevMap):
        for i in range(self.gmaps[0].height):
            for j in range(self.gmaps[0].width):
                xi, yi = self.gmaps[0].PixelToMeter(i, j)
                value = self.inverse_range_sensor_model(xi, yi)
                prevMap[i, j, 0] = np.clip(prevMap[i, j, 0] + value, 0, 255)
        return prevMap


    def resample(self):
        #if the robot is not moving, we do not resample
        amountOfMovement = la.norm(self.currOdom[0:1]-self.lastOdomLocalization[0:1])
        # print('amount of movement = {}'.format(amountOfMovement))
        if amountOfMovement < self.movement_thresh:
            return

        # computing the weights
        # poses_probs = np.zeros((self.N, ))
        # for i, xt in enumerate(self.currPoses):
        #     poses_probs[i] = self.measurement_model(xt, imageMap)

        #normalizing the weights
        print('resampling')
        # print(poses_probs.max(), poses_probs.min())
        #resampling
        sampledParticlesIndices = np.random.choice(range(self.N), size=self.N, p=self.weights)
        self.currPoses = self.currPoses[sampledParticlesIndices]
        self.weights = self.weights[sampledParticlesIndices]
        self.gmaps = self.gmaps[sampledParticlesIndices]

        self.lastOdomLocalization = self.currOdom
        
    def measurement_model(self, xt, imageMap):
        i, j = self.gmaps[0].metersToPixelsIndex(xt)
        Zt_star = ray_cast(i, j, imageMap, self.K_scanAngles + xt[2], self.Zmin, self.Zmax, debug=True)
        P_Ztk = 1
        for i, k in enumerate(self.K_scanAngles_indices):
            if self.Zt[k] == self.Zmax or Zt_star[i] == self.Zmax:
                continue
            probs = self.calcProbabilities(self.Zt[k], Zt_star[i])
            probs = probs*2
            p = np.dot(self.prob_weights, probs)
            P_Ztk = P_Ztk * p    
        return P_Ztk

    def sample_particles(self):
        self.weights = np.zeros((self.N, ))
        for k in range(self.N):
           xt = self.currPoses[k]
           self.weights[k] = self.measurement_model(xt, self.gmaps[k].map)
           self.gmaps[k].map = self.update_occupancy_grid(xt, self.gmaps[k].map)
        self.weights = self.weights/self.weights.sum()



def main():

    rospy.init_node('fast_slam', anonymous=True)
    robot = Robot(4)
    rospy.Subscriber('/gazebo/link_states', LinkStates, robot.linkStatesCallback)
    time.sleep(0.01)
    rospy.Subscriber('/husky_velocity_controller/odom', Odometry, robot.odometryCallback)
    rospy.Subscriber('/scan', LaserScan, robot.laserScanCallback)


    time.sleep(0.1)

    while not rospy.is_shutdown():
            
        Ws = robot.sample_particles()
        robot.resample()

        for i in range(robot.N):
            robot.plotZt(robot.gmaps[i].map, robot.currPoses[i])
            cv2.imshow('maps[{}]'.format(i),robot.gmaps[i].map)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break




if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
