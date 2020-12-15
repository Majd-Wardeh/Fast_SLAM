import cv2
import numpy as np
from numpy import pi, cos, sin
import numpy.linalg as la
from math import sqrt

from Queue import PriorityQueue

class Node:
    width = 211
    def __init__(self, point, parent, G_value, H_value):
        self.point = point
        self.parent = parent
        self.G = G_value
        self.H = H_value
        self.priority_value = G_value + H_value

        self.id = point[0]*self.width + point[1]

    def __lt__(self, other):
        return self.priority_value < other.priority_value

    def __str__(self):
        return '({}, {})'.format(self.point[0], self.point[1])

class Graph:
    def __init__(self, image, start_point, goal_point):
        self.image = image
        self.start_point = start_point
        self.goal_point = goal_point
        self.h, self.w = image.shape

        assert self.feasible(start_point), 'the start point is not feasible.'
        assert self.feasible(goal_point), 'the start point is not feasible.'

        self.openList = PriorityQueue()
        start_node = Node(start_point, None, 0, self.dis(start_point, goal_point) )
        self.openList.put(start_node)
        self.closeSet = set()
    
    def find_path(self):
        while not self.openList.empty():
            node = self.openList.get()
            self.closeSet.add(node.id)
            print(node)
            print(node.id)
            print(self.closeSet)
            _ = raw_input('...')
            if (node.point == self.goal_point).all():
                print('we have reached the goal!')
                print('the distance is ', node.G)
            else:
                self.update_openList(node)
            
    def dis(self, S, G):
        return la.norm(S-G)

    def update_openList(self, parentNode):
        x, y = parentNode.point
        points = np.array([[x+1, y], [x-1, y], [x, y+1], [x, y-1], [x+1, y+1], [x-1, y-1], [x-1, y+1], [x+1, y-1]])
        for p in points:
            if p[0] < 0 or p[0] >= self.w or p[1] < 0 or p[1] > self.h:
                print('outside the array bounds', p)
                continue
            if self.image[p[0], p[1]] > 0:
                print('obstacle p', self.image[p[0], p[1]])
                continue
            p_id = self.calc_id(p)
            if p_id not in self.closeSet:
                H = self.dis(p, self.goal_point)
                G = parentNode.G + self.dis(parentNode.point, p)
                node = Node(p, parentNode, G, H)
                self.openList.put(node)

    def feasible(self, p):
        condition1 = p[0] < 0 or p[0] >= self.w or p[1] < 0 or p[1] > self.h
        return condition1 == False and self.image[p[0], p[1]] == 0
    
    def calc_id(self, point):
        width = 211
        return point[0]*width + point[1]


rgb_image = cv2.imread("/home/majd/AUB/Mobile Robots/project/catkin_ws/src/fast_slam/maps/map.jpg")

image = np.uint8(rgb_image[:, :, 0] > 128)*255



bw_image = np.zeros((200, 200), dtype=np.uint8)
bw_image[80:100, 80:100] = 255
bw_image[30:50, :] = 255
kernel1 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0] ], dtype=np.uint8)
kernel2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1] ], dtype=np.uint8)

# erosion1 = cv2.erode(image,kernel1, iterations = 1)
# erosion2 = cv2.erode(image,kernel2, iterations = 1)

# updated_image1 = cv2.dilate(image, kernel1, iterations=5)
img = cv2.dilate(image, kernel2, iterations=5)


S = np.array([80, 180])
G = np.array([150, 50])

graph = Graph(img, S, G)

print(graph.dis(S, G))

graph.find_path()


rgb_image[:, :, 1] = 0
rgb_image[:, :, 2] = 0
rgb_image[80, 180, 2] = 255
rgb_image[150, 50, 2] = 255

# cv2.imshow('rgb', rgb_image)
# cv2.imshow("image", image)
# cv2.imshow("updated image2", img)
# cv2.waitKey(0)
