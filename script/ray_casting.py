import cv2
import numpy as np
from numpy import pi, cos, sin
from math import sqrt


def ray_cast(xco, yco, image, Zmin=1.0, Zmax=30.0, pixelToMeterRatio=0.1, debug=False):
    # xco with w, x the horiontal axis pointing right
    # y with h, y the virtical axis pointing up
    
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



# image = cv2.imread("/home/majd/AUB/Mobile Robots/project/catkin_ws/src/fast_slam/maps/map.jpg")
# image = image[:, :, 0]

image = np.zeros((201, 201, 3), dtype=np.uint8)
margin = 1
image[0:margin, :, 0] = 255
image[-margin:, :, 0] = 255
image[:, 0:margin, 0] = 255
image[:, -margin:, 0] = 255

xc = 200
yc = 200

# image = image[:, :, 0]
ray_cast(xc, yc, image, debug=True)

# cv2.circle(image, (xc, yc), radius=2, color=(0, 0, 255), thickness=-1)

cv2.imshow("image", image)
cv2.waitKey(0)
