# Updated upto 4/19/2021

"""
INFORMED_RRT_STAR 2D
@author: huiming zhou
"""

from cmath import inf
from email import iterators
import os
import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import matplotlib.patches as patches
import timeit
from scipy.spatial import distance
from matplotlib.patches import Ellipse


# sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
#                 "/../../Sampling_based_Planning/")

# from Sampling_based_Planning.rrt_2D import env, plotting, utils

import env, plotting, utils

ellipseList = []
totalEllipseNodes = []
numberOfGoalChanges = 0
currentEllipse = []
nodeInEllipse = []
indivdiualEllipseNode = []
overlapPointInEllipse = set()
ellipseFoundIerationNo = 0

class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class IRrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max):
        self.x_start = Node(x_start)
        self.x_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.fig, self.ax = plt.subplots()
        self.delta = self.utils.delta
        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

        self.V = [self.x_start]
        self.X_soln = set()
        self.path = None

    def init(self):
        cMin, theta = self.get_distance_and_angle(self.x_start, self.x_goal)
        C = self.RotationToWorldFrame(self.x_start, self.x_goal, cMin)
        xCenter = np.array([[(self.x_start.x + self.x_goal.x) / 2.0],
                            [(self.x_start.y + self.x_goal.y) / 2.0], [0.0]])
        x_best = self.x_start

        return theta, cMin, xCenter, C, x_best

    
    def planning(self):
        theta, dist, x_center, C, x_best = self.init()
        c_best = np.inf
        
        ##online IRRT* valriable
        isEllipseInserted = False
        singleEllipsePointSet = []
        print("isEllipseInserted started ",isEllipseInserted)        
        for k in range(self.iter_max):                        
            # for node in self.V:
            #     print("node ",node.x,node.y)
            # print("ellipseFoundIerationNo ",ellipseFoundIerationNo)
            # if len(ellipseList) == 2:
            #     # print("ellipseCnt ",ellipseList)
            #     a = math.sqrt(ellipseList[0][1] ** 2 - ellipseList[0][2] ** 2) / 2.0
            #     b = ellipseList[0][1] / 2.0
            #     areaE1 = 3.14 * a * b
            #     print(areaE1)                
            #     density = len(indivdiualEllipseNode[0])/areaE1
            #     a = math.sqrt(ellipseList[1][1] ** 2 - ellipseList[1][2] ** 2) / 2.0
            #     b = ellipseList[1][1] / 2.0
            #     areaE2 = 3.14 * a * b
            #     print(" areaE2 ", areaE2)
            #     print(density)
            #     print("ellipseFoundIerationNo ", ellipseFoundIerationNo)
            #     print("(density * areaE2) ", (density * areaE2))
            #     ellipseFoundIerationNo = math.floor(ellipseFoundIerationNo + (density * areaE2))
            #     self.iter_max = ellipseFoundIerationNo
            #     print("ellipseFoundIerationNo ", ellipseFoundIerationNo)

            if self.X_soln:
                cost = {node: self.Cost(node) for node in self.X_soln}
                x_best = min(cost, key=cost.get)
                c_best = cost[x_best]

                # print("c_best ",c_best," dist ",dist," x_center ",x_center," C ",C)
            x_rand = self.Sample(c_best, dist, x_center, C)
            x_nearest = self.Nearest(self.V, x_rand)
            x_new = self.Steer(x_nearest, x_rand)

            if k % 500 == 0:
                print(k)
                # print(x_new.parent," ",x_new.x," ",x_new.y)

            if x_new and not self.utils.is_collision(x_nearest, x_new):
                X_near = self.Near(self.V, x_new)
                c_min = self.Cost(x_nearest) + self.Line(x_nearest, x_new)
                # print("EllipseList ",ellipseList)
                iterationCnt = len(ellipseList) - 1
                if iterationCnt <= 0:
                    self.V.append(x_new)
                # print("iterationCnt ",iterationCnt)
                allPreviousEllipseList = ellipseList[:len(ellipseList)-1]
                # iterator = 0;
                for iterator in allPreviousEllipseList:
                    # if cnt < iterationCnt :                         
                        ellipseCnt = iterator
                        # print("ellipseCnt ",ellipseCnt)                     
                        xx_center = ellipseCnt[0]
                        h = xx_center[0]
                        k = xx_center[1]
                        cc_best = ellipseCnt[1]
                        ddist = ellipseCnt[2]
                        # print(xx_center)
                        # print(cc_best)
                        # print(ddist)
                        a = math.sqrt(cc_best ** 2 - ddist ** 2) / 2.0
                        b = c_best / 2.0
                        x= x_new.x
                        y= x_new.y
                        p = ((math.pow((x - h), 2) // math.pow(a, 2)) + (math.pow((y - k), 2) // math.pow(b, 2)))
                        # print("P value ",p)
                        if p > 1:                    
                            # print("isPointInEllispeOutside ")
                            self.V.append(x_new)
                        else:                
                            closestDist = inf
                            previousEllipseIndex = ellipseList.index(iterator)                                           
                            previousEllipse = indivdiualEllipseNode[previousEllipseIndex]
                            # print("previousEllipse ",previousEllipse)
                            x_newList = []
                            x_newList.append([x_new.x,x_new.y])
                            for nodes in previousEllipse:                                                                
                                #  print("x_newList ",x_newList[0][1],x_newList[0][1])
                                 nodesList = []
                                 nodesList.append([nodes.x,nodes.y])                                
                                #  print("nodesList ",nodesList[0][0],nodesList[0][1])
                                 tmpDist = distance.euclidean(x_newList,nodesList)
                                #  print("distance ",tmpDist)
                                 if closestDist > tmpDist:
                                    closestDist = tmpDist
                                    x_tmp = nodes
                                    # print("x_tmp ",x_tmp.x)
                            overlapPointInEllipse.add(x_tmp)        
                            x_new = x_tmp
                            # print("previous node in ellipse ",x_new.x,x_new.y)   
                        # iterator = iterator + 1 
                # self.V.append(x_new)                
                # totalEllipseNodes.append(x_new)

                # choose parent
                for x_near in X_near:
                    c_new = self.Cost(x_near) + self.Line(x_near, x_new)
                    if c_new < c_min:
                        x_new.parent = x_near
                        c_min = c_new

                # rewire
                for x_near in X_near:
                    c_near = self.Cost(x_near)
                    c_new = self.Cost(x_new) + self.Line(x_new, x_near)
                    if c_new < c_near:
                        x_near.parent = x_new

                if self.InGoalRegion(x_new):
                    # print("InGoalRegion ",x_new)'
                    # print("InGoalRegion ",self.InGoalRegion(x_new));
                    if not self.utils.is_collision(x_new, self.x_goal):
                        self.X_soln.add(x_new)
                        # new_cost = self.Cost(x_new) + self.Line(x_new, self.x_goal)
                        # if new_cost < c_best:
                        #     c_best = new_cost
                        #     x_best = x_new
            
            if c_best != np.inf:
                # print("K--> ",k)
                ellipseTmpList = []
                # print("isEllipseInserted ",isEllipseInserted)
                if isEllipseInserted:
                    # print("K--> ",k)
                    singleEllipsePointSet.append(x_new)
                # print("singleEllipsePointSet node ",x_new.x)
                if self.InGoalRegion(x_new) and not isEllipseInserted:
                    isEllipseInserted = True
                    ellipseTmpList.append(x_center)
                    ellipseTmpList.append(c_best)
                    ellipseTmpList.append(dist)
                    ellipseTmpList.append(theta)
                    currentEllipse.append(ellipseTmpList)
                    ellipseList.append(ellipseTmpList) 
                    ellipseFoundIerationNo = k  
                    # print("EllipseList ",ellipseTmpList)
                    # for i in ellipseList:
                    #     print(i)
                    # print("EllipseList ",ellipseList[])
                    # self.animation(x_center=ellipseList[0][0],c_best=ellipseList[0][1],dist=ellipseList[0][2],theta=ellipseList[0][3])
                    # self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)

        self.path = self.ExtractPath(x_best)        
        print("overlapPointInEllipse ",len(overlapPointInEllipse))
        # print(singleEllipsePointSet)
        print("singleEllipsePointSet ",len(singleEllipsePointSet))
        indivdiualEllipseNode.append(singleEllipsePointSet)
        # print("indivdiualEllipseNode ", indivdiualEllipseNode)
        self.animation(x_center=x_center, c_best=c_best, dist=dist, theta=theta)
        plt.plot([x for x, _ in self.path], [y for _, y in self.path], '-r')
        plt.pause(0.01)
        plt.show()
        # print("Self.V size ",len(self.V))
        # print("parent ",self.parent)
        # for node in self.V:
        #     print("Node ",node.x," ",node.y)
        return self.path

    def Steer(self, x_start, x_goal):
        dist, theta = self.get_distance_and_angle(x_start, x_goal)
        # print("Dist ",dist)
        dist = min(self.step_len, dist)
        node_new = Node((x_start.x + dist * math.cos(theta),
                         x_start.y + dist * math.sin(theta)))
        node_new.parent = x_start

        return node_new

    def Near(self, nodelist, node):
        n = len(nodelist) + 1
        r = 50 * math.sqrt((math.log(n) / n))

        dist_table = [(nd.x - node.x) ** 2 + (nd.y - node.y) ** 2 for nd in nodelist]
        X_near = [nodelist[ind] for ind in range(len(dist_table)) if dist_table[ind] <= r ** 2 and
                  not self.utils.is_collision(nodelist[ind], node)]

        return X_near

    def Sample(self, c_max, c_min, x_center, C):
        if c_max < np.inf:
            r = [c_max / 2.0,
                 math.sqrt(c_max ** 2 - c_min ** 2) / 2.0,
                 math.sqrt(c_max ** 2 - c_min ** 2) / 2.0]
            L = np.diag(r)
            # print("r ",r," L ",L)

            while True:
                x_ball = self.SampleUnitBall()
                x_rand = np.dot(np.dot(C, L), x_ball) + x_center
                if self.x_range[0] + self.delta <= x_rand[0] <= self.x_range[1] - self.delta and \
                        self.y_range[0] + self.delta <= x_rand[1] <= self.y_range[1] - self.delta:
                    break
            x_rand = Node((x_rand[(0, 0)], x_rand[(1, 0)]))
        else:
            # print("c_max < np.inf ",c_max," ",np.inf)
            x_rand = self.SampleFreeSpace()

        return x_rand

    @staticmethod
    def SampleUnitBall():
        while True:
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x ** 2 + y ** 2 < 1:
                return np.array([[x], [y], [0.0]])

    def SampleFreeSpace(self):
        delta = self.delta

        if np.random.random() > self.goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.x_goal

    def ExtractPath(self, node):
        path = [[self.x_goal.x, self.x_goal.y]]

        while node.parent:
            path.append([node.x, node.y])
            node = node.parent

        path.append([self.x_start.x, self.x_start.y])

        return path

    def InGoalRegion(self, node):
        if self.Line(node, self.x_goal) < self.step_len:
            return True

        return False

    @staticmethod
    def RotationToWorldFrame(x_start, x_goal, L):
        a1 = np.array([[(x_goal.x - x_start.x) / L],
                       [(x_goal.y - x_start.y) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    @staticmethod
    def Nearest(nodelist, n):
        return nodelist[int(np.argmin([(nd.x - n.x) ** 2 + (nd.y - n.y) ** 2
                                       for nd in nodelist]))]

    @staticmethod
    def Line(x_start, x_goal):
        return math.hypot(x_goal.x - x_start.x, x_goal.y - x_start.y)

    def Cost(self, node):
        if node == self.x_start:
            return 0.0

        if node.parent is None:
            return np.inf

        cost = 0.0
        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def animation(self, x_center=None, c_best=None, dist=None, theta=None):
        plt.cla()
        self.plot_grid("Online Informed rrt*, N = " + str(self.iter_max))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for node in self.V:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

        if c_best != np.inf:
            self.draw_ellipse(x_center, c_best, dist, theta)

        plt.pause(0.01)

    def plot_grid(self, name):

        for (ox, oy, w, h) in self.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        plt.plot(self.x_start.x, self.x_start.y, "bs", linewidth=3)
        plt.plot(self.x_goal.x, self.x_goal.y, "rs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    @staticmethod
    def draw_ellipse(x_center, c_best, dist, theta):
        a = math.sqrt(c_best ** 2 - dist ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        # print("Ellipse",cx,cy,a,b)
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, ".b")
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)
            
    # def isPointInEllispeOutside(self,h, k, x, y, a, b):
    #     p = ((math.pow((x - h), 2) // math.pow(a, 2)) +
	# 	(math.pow((y - k), 2) // math.pow(b, 2)))
    #     if p > 1:
    #         return True
    #     return False
    def draw_ellipse1(x_center, c_best, dist, theta):
        a = math.sqrt(c_best ** 2 - dist ** 2) / 2.0
        b = c_best / 2.0
        angle = math.pi / 2.0 - theta
        cx = x_center[0]
        cy = x_center[1]
        # print("Ellipse",cx,cy,a,b)
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, ".b")
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2)

def main():        
    startTime = timeit.default_timer()
    x_start = (18, 8)  # Starting node
    x_goal = (37, 18)  # Goal node

    irrt_star = IRrtStar(x_start, x_goal, 1, 0.10, 20, 1500)
    path = irrt_star.planning()
    # print("Path ",path)
    path.reverse()
    # print("Path ",path)
    print("start and goal changed ",x_start," ",x_goal)
    totalPath = []    
    # goal = x_goal

    numberOfTimesGoalChanged = 1
    goalChangingTime = len(path)-3
    goalTmp = [[40,5],[43,10],[25,10]]
    i = 0 # for next wayPoint
    x = 0 # for next possible goal
    isGoalChanged = False
    while path[i] != x_goal:
      totalPath.append(path[i])
    #   print("I value ",i)
      currentEllipseNode = []
    #   print("currentEllispe ",currentEllipse)
    #   print("currentEllispe ",len(totalEllipseNodes))
      if currentEllipse:
        currentEllipseTmp = currentEllipse[0]
        for node in totalEllipseNodes:
            xx_center = currentEllipseTmp[0]
            h = xx_center[0]
            k = xx_center[1]
            cc_best = currentEllipseTmp[1]
            ddist = currentEllipseTmp[2]
            # print(xx_center)
            # print(cc_best)
            # print(ddist)
            a = math.sqrt(cc_best ** 2 - ddist ** 2) / 2.0
            b = cc_best / 2.0
            xx= node.x
            y= node.y
            p = ((math.pow((xx - h), 2) // math.pow(a, 2)) + (math.pow((y - k), 2) // math.pow(b, 2)))
            # print( "P ",p)
            if p <= 1:
                nodeInEllipse.append(node)
                currentEllipseNode.append(node)
      currentEllipse.clear()
      totalEllipseNodes.clear()
    #   print("CurrentEllipses node ",len(currentEllipseNode))
    #   print("Total nodes in all ellipse ",len(nodeInEllipse))
      currentEllipseNode.clear()

      if i == goalChangingTime:
        if x < numberOfTimesGoalChanged :
          x_goal = goalTmp[x] # goal changed
          isGoalChanged = True
          x_start = path[i]
          x += 1      

        #   print("start and goal changed ",x_start," ",x_goal)
          irrt_start = IRrtStar(x_start, x_goal, 10, 0.10, 20, 1500)
          path = irrt_start.planning()
          goalChangingTime = len(path)-3
          path.reverse()
        #   print("Changed Path ",path)          
      i += 1
      if isGoalChanged: 
        i = 0
        isGoalChanged = False
    #   print("Path changed ",path)  
      # print("Path after changed goal ",path[i]," new goal ",goal)    
    overlapPoint = []
    for node in nodeInEllipse:
        # print("node ",node.x," ",node.y)
        if nodeInEllipse.count(node) > 1:
            # print("Overlap Node ",node)
            overlapPoint.append(node) 
    # print("Overlap ",len(overlapPoint))
    totalPath.append(x_goal)
    totalPathDistance = 0
    # print(len(totalPath))
    for i in range(len(totalPath)-1):
      totalPathDistance += distance.euclidean(totalPath[i],totalPath[i+1])
      # print(totalPath[i])
    # for i in range(totalPath):
    #   totalPathDistance += distance.euclidean(totalPath[i],totalPath[i+1])    
    print("Total Path Distance ",totalPathDistance)
    # print("Start ",x_start,"Goal ",x_goal)
    # print("Path",path)
    # print("TotalPath ",totalPath)

    if path is None:
        print("Cannot find path")
    else:
        end = timeit.default_timer()
        print ('RRT* End : ', end)
        print ('RRT*  elapsed time is : ', end - startTime)
        print("found path!!")
        print("TotalellipseNodes ",len(totalEllipseNodes))
    # print("EllipseList ",ellipseList)   

        plt.cla()
        # plt.plot_grid("Informed rrt*, N = " + str(700))
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        for node in nodeInEllipse:
            if node.parent:
                plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")        
        # plt.pause(0.01)
    # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        cnt = 0
    for ellispeCnt in ellipseList:                    
        # ellipseNodesIndex = ellipseList.index(ellispeCnt)
        # print("ellipseNodesIndex ",ellipseNodesIndex)
        xx_center = ellispeCnt[0]
        c_best=ellispeCnt[1]
        dist=ellispeCnt[2]
        theta=ellispeCnt[3]
        print(xx_center[0])
        print(c_best)
        print(dist)
        print(theta)                    
        a = math.sqrt(c_best ** 2 - dist ** 2) / 2.0
        b = c_best / 2.0
        ellipseArea = 3.14 * a * b
        print("ellipseArea ",ellipseArea)
        print("Current Ellipse Nodes ", len(indivdiualEllipseNode[cnt]))
        print("Density of Ellipse ",cnt," is ",len(indivdiualEllipseNode[cnt])/ellipseArea)
        angle = math.pi / 2.0 - theta
        cx = xx_center[0]
        cy = xx_center[1]
        # print("Ellipse",cx,cy,a,b)
        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
        fx = rot @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        cnt = cnt + 1 
        print("cnt ",cnt)

        # ells= Ellipse((cx,cy),width=a, height=b,angle=angle,linewidth=2,alpha=0.5)
        # print("Ellipse ",ells)       
        # plt.plot(cx, cy, ".b")
        plt.plot(18,8,".r",linewidth=5)
        # plt.plot(x_start.x,x_start.y,".r",linewidth=5)
        plt.plot(37,18,".b",linewidth=5)
        plt.plot(40,5,".b",linewidth=5)
        plt.plot(px, py, linestyle='--', color='darkorange', linewidth=2) 
        # ax.add_artist(ells)
            

        # xx_center = ellispeCnt[0], cc_best=ellispeCnt[1], ddist=ellispeCnt[2], ttheta=ellispeCnt[3]
        # print(xx_center)
        # print(cc_best)
        # print(ddist)
        # print(ttheta)

    # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    # for ellispeCnt in ellipseList:
    #     x_center = ellispeCnt[0]
    #     c_best=ellispeCnt[1]
    #     dist=ellispeCnt[2]
    #     theta=ellispeCnt[3]
    #     # print(x_center)
    #     # print(c_best)
    #     # print(dist)
    #     # print(theta)
    #     a = math.sqrt(c_best ** 2 - dist ** 2) / 2.0
    #     b = c_best / 2.0
    #     angle = math.pi / 2.0 - theta
    #     cx = x_center[0]
    #     cy = x_center[1]
    #     # print("Ellipse",cx,cy,a,b)
    #     t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    #     x = [a * math.cos(it) for it in t]
    #     y = [b * math.sin(it) for it in t]
    #     rot = Rot.from_euler('z', -angle).as_matrix()[0:2, 0:2]
    #     fx = rot @ np.array([x, y])
    #     px = np.array(fx[0, :] + cx).flatten()
    #     py = np.array(fx[1, :] + cy).flatten()
    #     ells= Ellipse((cx[0],cy[0]),width=a, height=b,angle=angle,alpha=0.5)
    #     print(ells)
    #     ax.add_artist(ells)
    # ells = [Ellipse((i[0][0],i[0][1]),
    #             width=i[1], height=i[2],
    #             angle=math.pi / 2.0 - i[3],linewidth=2,alpha=0.5)
    #     for i in ellipseList]    
    # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    # angle_step = 45  # degrees
    # angles = np.arange(0, 180, angle_step)

    # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    # for angle in angles:
    #     ellipse = Ellipse((0, 0), 4, 2, angle=angle, alpha=0.5)
    #     ax.add_artist(ellipse)

    # ax.set_xlim(-2.2, 2.2)
    # ax.set_ylim(-2.2, 2.2)




    # for e in ells:
    #     ax.add_artist(e)
    #     # e.set_clip_box(ax.bbox)
    #     # e.set_alpha(np.random.rand())
    #     # e.set_facecolor(np.random.rand(3))

    # ax.set_xlim(0, 50)
    # ax.set_ylim(-5, 35)

    plt.show()

if __name__ == '__main__':
    main()
