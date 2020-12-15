from __future__ import print_function
import cv2
import numpy as np
from numpy import pi, cos, sin
import numpy.linalg as la
from math import sqrt

from Queue import PriorityQueue

class Node:
       
    def __init__(self, point, parent, G_value, H_value, width=211):
        self.point = point
        self.parent = parent
        self.G = G_value
        self.H = H_value
        self.priority_value = G_value + H_value

        self.id = point[0]*width + point[1]

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
        self.debug_image = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.debug_image[:, :, 0] = self.image
        self.debug_image[start_point[0], start_point[1], 2] = 255
        self.debug_image[goal_point[0], goal_point[1], 2] = 255

        assert self.feasible(start_point), 'the start point is not feasible.'
        assert self.feasible(goal_point), 'the start point is not feasible.'

        self.openList = PriorityQueue()
        start_node = Node(start_point, None, 0, self.dis(start_point, goal_point) )
        self.openList.put(start_node)
        self.closeSet = set()
    
    def find_path(self, debug=False):
        while not self.openList.empty():
            node = self.openList.get()
            if node.id in self.closeSet:
                continue
            self.closeSet.add(node.id)

            if debug:
                self.debug_image[node.point[0], node.point[1], 1] = 255
                cv2.imshow('debug image', self.debug_image)
                cv2.waitKey(1)

            # print('{} and node.Id = {}'.format(node, node.id))

            if (node.point == self.goal_point).all():
                # print('we have reached the goal!')
                # print('the distance is ', node.G)
                break
            else:
                self.update_openList(node)

        if debug:
            self.debug_image[:, :, 1] = np.uint8(self.debug_image[:, :, 1]*0.3)
        path = []
        while node != None:
            path.append(node.point)
            if debug:
                self.debug_image[node.point[0], node.point[1], 2] = 255
            node = node.parent

        if debug:    
            cv2.imshow('debug image', self.debug_image)
            cv2.waitKey(1)
        path = np.array(path)
        return np.flip(path, 0)

            
    def dis(self, S, G):
        return la.norm(S-G)

    def update_openList(self, parentNode):
        # print('pnode: {} had neighborhoods: '.format(parentNode), end='')
        x, y = parentNode.point
        points = np.array([[x+1, y], [x-1, y], [x, y+1], [x, y-1], [x+1, y+1], [x-1, y-1], [x-1, y+1], [x+1, y-1]])
        for p in points:
            if p[0] < 0 or p[0] >= self.w or p[1] < 0 or p[1] > self.h:
                # print('outside the array bounds', p)
                continue
            if self.image[p[0], p[1]] > 0:
                # print('obstacle p', self.image[p[0], p[1]])
                continue
            p_id = self.calc_id(p)
            if p_id not in self.closeSet:
                H = self.dis(p, self.goal_point)
                G = parentNode.G + self.dis(parentNode.point, p)
                node = Node(p, parentNode, G, H)
                self.openList.put(node)
                # print('{}, '.format(node), end='')
        # print()

    def feasible(self, p):
        condition1 = p[0] < 0 or p[0] >= self.w or p[1] < 0 or p[1] > self.h
        return condition1 == False and self.image[p[0], p[1]] == 0
    
    def calc_id(self, point):
        width = 211
        return point[0]*width + point[1]


def main():
    rgb_image = cv2.imread("/home/majd/AUB/Mobile Robots/project/catkin_ws/src/fast_slam/maps/map.jpg")

    image = np.uint8(rgb_image[:, :, 0] > 128)*255
    rgb_image[:, :, 1] = 0
    rgb_image[:, :, 2] = 0

    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0] ], dtype=np.uint8)
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1] ], dtype=np.uint8)
    img = cv2.dilate(image, kernel, iterations=5)

    Start = np.array([80, 180])
    Goal = np.array([150, 50])
    # Goal = np.array([20, 20])

    graph = Graph(img, Start, Goal)
    path = graph.find_path(debug=True)

    cv2.circle(rgb_image, (Start[1], Start[0]), radius=5, color=(0, 255, 0), thickness=1)
    cv2.circle(rgb_image, (Goal[1], Goal[0]), radius=5, color=(0, 0, 255), thickness=1)
    for point in path:
        rgb_image[point[0], point[1], 2] = 255

    cv2.imshow('map', rgb_image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()