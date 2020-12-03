import cv2
import numpy as np
from numpy import pi, cos, sin

image = cv2.imread("/home/majd/AUB/Mobile Robots/project/catkin_ws/src/fast_slam/maps/map.jpg")
image = image[:, :, 0]

xc = 100
yc = 100

MatrixSize = image.shape

r = np.linspace(0, 30,10)
n = 10
thetastep = 2*pi/n
for i in range(n):
    theta = thetastep * i
    print(theta)
    print((r*cos(theta)))
    x = xc + r*cos(theta)
    y = yc + r*sin(theta)
print(x)
print(y)
    # Removing points out of map
    tempX = []
    print('x.shape = ', x.shape)
    for k in range(x.shape[0]):
        if x[k] > MatrixSize[1] or x[k] <= 0:
            tempX.append(k)

    tempY = []
    print('y.shape = ', y.shape)
    for k in range(y.shape[0]):
        if y[k] > MatrixSize[0] or y[k] <= 0: 
            tempY.append(k)    
    # x(temp)=[]
    # y(temp)=[]




# cv2.imshow("image", image)
# cv2.waitKey(0)
