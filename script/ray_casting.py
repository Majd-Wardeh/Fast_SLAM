import cv2
import numpy as np
from numpy import pi, cos, sin
from math import sqrt


def ray_cast(xc, yc, image, pixelToMeterRatio=0.1):
    # x with w, x the horiontal axis
    # y with h
    h, w, _ = image.shape
    yc = h-1 - yc



    r = np.linspace(0, 200, 1000)
    n = 8
    thetastep = 2*pi/n
    ranges = np.zeros((n,))
    angles = np.zeros((n,))
    for i in range(n):
        theta = thetastep * i
        angles[i] = theta
        print('theta =',theta)
        x = xc + r*cos(theta)
        y = yc + r*sin(theta)
    
        xint = np.int32(np.round(x))
        yint = np.int32(np.round(y))

        index = np.logical_and(np.logical_and(xint < w, xint >= 0), np.logical_and(yint < h, yint >= 0))
        xint = xint[index]
        yint = yint[index]

        intersection_index = np.argmax(image[xint, yint, 0])
        px, py = xint[intersection_index], yint[intersection_index]
        ranges[i] = sqrt( (px - xc)**2 + (py - yc)**2  )
        print(intersection_index)

        cv2.circle(image, (py, px), radius=2, color=(0, 0, 255), thickness=-1)
        image[xint, yint, 1] = 255
    print('------------------------')
    ranges = ranges*pixelToMeterRatio
    print(angles)
    print(ranges)
    cv2.imshow('image', image)
    cv2.waitKey(0)



# image = cv2.imread("/home/majd/AUB/Mobile Robots/project/catkin_ws/src/fast_slam/maps/map.jpg")
# image = image[:, :, 0]

image = np.zeros((201, 201, 3), dtype=np.uint8)
margin = 8
image[0:margin, :, 0] = 255
image[-margin:, :, 0] = 255
image[:, 0:margin, 0] = 255
image[:, -margin:, 0] = 255

xc = 10
yc = 180

ray_cast(xc, yc, image)





# cv2.imshow("image", image)
# cv2.waitKey(0)
