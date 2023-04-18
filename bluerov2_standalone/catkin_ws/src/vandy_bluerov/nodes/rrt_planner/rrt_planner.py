#!/usr/bin/env python

# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for VU in 2020

# TODO clear out unnecessary imports
import numpy as np
import math
from scipy.spatial.transform import Rotation
from multiprocessing import Process

from math import sqrt
import random
from quads import QuadTree, BoundingBox
import copy

import math
import rospy
# import heading
import math
from scipy.spatial.transform import Rotation

from uuv_control_msgs.srv import *
from std_msgs.msg import String, Time, Float64MultiArray, Bool, Int32
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from vandy_bluerov.msg import HSDCommand
import time

# Gets the yaw Euler angle from a quaternion in the form [x, y, z, w]
# Returns the yaw from the conversion
def get_yaw_from_quaternion(Q):
    return Rotation.from_quat([Q[0], Q[1], Q[2], Q[3]]).as_euler('XYZ')[2]

# Gets the rotation matrix from a quaternion in the form [x, y, z, w]
# Returns the rotation matrix from the conversion
def get_rotation_matrix_from_quaternion(Q):
    return Rotation.from_quat([Q[0], Q[1], Q[2], Q[3]]).as_dcm()

# Normalizes a radians angle to be between [-PI, PI]
# Returns the normalized yaw
def normalize(theta):
    while theta > math.pi:
        theta -= 2*math.pi
    while theta < -math.pi:
        theta += 2*math.pi

    return theta

# Converts x, y, theta from NED to ENU coordinate system
# Returns x, y, and theta in the ENU system
def NED_to_ENU(x, y, theta):    
    return y, x, normalize(math.pi / 2 - theta)

# Converts x, y, theta from ENU to NED coordinate system
# Returns x, y, and theta in the NED system
def ENU_to_NED(x, y, theta):    
    return y, x, normalize(math.pi / 2 - theta)

# Our local occupancy grid representation as a numpy matrix
class RRTOccupancyGrid:
    # Takes in the ros OccupancyGrid message and converts it into our form
    # Our occupancy grid is held in the variable self.grid and uses the ENU system for positions
    def __init__(self, occupancy_grid_message):
        self.width = occupancy_grid_message.info.width # (cells)
        self.height = occupancy_grid_message.info.height # (cells)
        self.resolution = occupancy_grid_message.info.resolution # (meters/cell)
        data = occupancy_grid_message.data
        
        yaw = get_yaw_from_quaternion([occupancy_grid_message.info.origin.orientation.x,occupancy_grid_message.info.origin.orientation.y,occupancy_grid_message.info.origin.orientation.z,occupancy_grid_message.info.origin.orientation.w])
        self.world_to_occupancy = np.array([[math.cos(yaw), -math.sin(yaw), occupancy_grid_message.info.origin.position.x],[math.sin(yaw), math.cos(yaw), occupancy_grid_message.info.origin.position.y],[0.0,0.0,1.0]])
        self.occupancy_to_world = np.linalg.inv(self.world_to_occupancy)

        self.grid = np.array(data).reshape((self.height, self.width))
        new_grid = np.empty((self.height, self.width, 2), dtype=Bool)
        new_grid[np.where(self.grid != 0)] = np.array([True, False]) # obstacle
        new_grid[np.where(self.grid == 0)] = np.array([False, False]) # free
        self.grid = new_grid
    
    # Converts a coordinate from world ENU system to our occupancy grid system (defined by the lower left corner) 
    def grid_transform(self, world_x, world_y):
        occupancy_x = np.matmul(self.occupancy_to_world, np.array([[world_x],[world_y],[1]]))[0,0]
        occupancy_y = np.matmul(self.occupancy_to_world, np.array([[world_x],[world_y],[1]]))[1,0]
        return occupancy_x, occupancy_y

    # Get the coordinates in world frame in meters of the four coordinates of the occupancy grid
    # returns the maximized boundaries for the occupancy grid in the form [min_x, max_x, min_y, max_y]
    # NOTE: does not account for rotation, if it is rotated it will return the the minimum possible and maximum possible
    def get_occupancy_boundaries(self):
        bounds = np.array([[0,0,self.width*self.resolution,self.width*self.resolution],
                           [0,self.height*self.resolution,0,self.height*self.resolution],
                           [1,1,1,1]])
        world_bounds = np.transpose(np.matmul(self.world_to_occupancy, bounds)[0:2])
        return [np.min(world_bounds[:,0]), np.max(world_bounds[:,0]), np.min(world_bounds[:,1]), np.max(world_bounds[:,1])]
    
    # Queries the occupancy grid for ENU x and y world coordinates in meters
    # Returns true if there is an obstacle, returns false if there is not
    def query_obstacle(self, x, y):
        x, y = self.grid_transform(x, y)
        return self.grid[int(y // self.resolution), int(x // self.resolution), 0]
    
    # Queries the occupancy grid for ENU x and y world coordinates in meters
    # Returns true if it is free, returns false if it is not
    def query_free(self, x, y):
        x, y = self.grid_transform(x, y)
        return not self.grid[int(y // self.resolution), int(x // self.resolution), 0]

    # Queries the occupancy grid for a path (list of Nodes)
    # Returns true if the entire path is free, returns false if the path encounters an obstacle
    def query_path_free(self, path):
        if path != None and len(path) >= 2:
            for i in range(len(path) - 1):
                if not self.query_free(path[i].pos.x, path[i].pos.y, path[i+1].pos.x, path[i+1].pos.y):
                    return False

        return True

    # Queries the occupancy grid for a line between two points (x1,y1) and (x2,y2)
    # Returns true if the entire line is free, returns false if the line encounters an obstacle
    def query_free(self, x1, y1, x2, y2):
        x1, y1 = self.grid_transform(x1, y1)
        x2, y2 = self.grid_transform(x2, y2)
        x1 /= self.resolution
        y1 /= self.resolution
        x2 /= self.resolution
        y2 /= self.resolution
        # make sure x1 is on the left
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        # y = ax + b
        a = (y2 - y1) / (x2 - x1)
        # checks if line is vertical
        if math.isnan(a):
            if y2 > y1:
                a = 10 ** 10
            else:
                a = -10 ** 10
        b = y1 - x1 * a
        start = int(x1)
        end = int(x2 + 1)
        
        if y1 < y2:
            for index, x_left in enumerate(range(start, end)):
                x_right = x_left + 1
                y_start = int(a*x_left+b)
                y_end = int(a*x_right+b)
                if index == 0:
                    y_start = int(y1)
                if index == end - start - 1:
                    y_end = int(y2)
                y_list = list(range(y_start, y_end+1))
                for y in y_list:
                    if x_left >= self.grid.shape[0] or x_left < 0 or y >= self.grid.shape[1] or y < 0:
                        continue
                    if self.grid[y, x_left, 0]:
                        return False
        
        if y1 >= y2:
            for index, x_left in enumerate(range(start, end)):
                x_right = x_left + 1
                y_start = int(a*x_left+b)
                y_end = int(a*x_right+b)
                if index == 0:
                    y_start = int(y1)
                if index == end - start - 1:
                    y_end = int(y2)
                y_list = list(range(y_start, y_end-1, -1))
                for y in y_list:
                    if x_left >= self.grid.shape[0] or x_left < 0 or y >= self.grid.shape[1] or y < 0:
                        continue
                    if self.grid[y, x_left, 0]:
                        return False
        return True

# Class representing an RRTPoint (which holds an x, y, and theta)
class RRTPoint():
    # Initializes the RRTPoint
    def __init__(self, x, y, theta = 0):
        self.x = x
        self.y = y
        self.theta = theta

    # Checks two points for equality
    # Returns true if the two points are the same. False otherwise
    def __eq__(self, point):
        return self.x == point.x and self.y == point.y and self.theta == point.theta
    
    # Converts a point to a string for easier debugging
    # Returns the string of the point
    def __str__(self):
        return str(self.x) + ' ' + str(self.y) + ' ' + str(self.theta)
    
    # Adds all the values from two points
    def add(self, point):
        self.x += point.x
        self.y += point.y
        self.theta += point.theta
        
    # Gets the distance of a point from the origin (magnitude as if it was a vector)
    # Returns the distance
    def mag(self):
        return sqrt((self.x) ** 2 + (self.y) ** 2)
    
    # Get the angle from the self point towards the parameter point
    # Returns the angle in radians
    def get_angle_towards(self, point):
        return math.atan2(point.y - self.y, point.x - self.x)

    # Get the distance between two points
    # Returns the distance between the two points
    def distance(self, point):
        return sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

# Class representing a node (vertex) of our RRT Tree
class Node():
    # Initializes a node with its position (RRTPoint), and an optional cost value
    def __init__(self, point, cost = None):
        self.pos = point # Position of the point
        self.cost = cost # RRT* "cost" of the point - represents the distance from the starting node through the tree to get to that node
        self.parent = None # The Node parent of this node in the tree
        self.children = [] # A list of Node children of this node in the tree # TODO consider making this a dictionary to speed up lookups
        self.node_index = -1 # The index of this node in the dictionary holding nodes for the tree
        
    # Gets the x value of the node for the QuadTree
    # Returns the x value
    def get_x(self):
        return self.pos.x
    
    # Gets the y value of the node for the QuadTree
    # Returns the y value
    def get_y(self):
        return self.pos.y
        
    # Converts a Node to a string for easier debugging
    # Returns the string of the Node
    def __str__(self):
        s = str(self.node_index) + ': ' + str(self.pos) + ' | Cost:' + str(self.cost)
        return s
    
    # Gets the distance between two Nodes
    # Returns the distance between two Nodes
    def distance(self, node2):
        return sqrt((self.pos.x - node2.pos.x) ** 2 + (self.pos.y - node2.pos.y) ** 2)

# Class representing the tree of our RRT* algorithm
class Tree():
    # Initialize the tree with the given parameters
    def __init__(self, radius, max_num_nodes, world_bounds_inflation):
        # PARAMS
        self.radius = radius # the connection radius for the RRT*
        self.max_num_nodes = max_num_nodes # the maximum number of nodes to have in the tree
        self.world_bounds_inflation = world_bounds_inflation # how much to inflate the world bounds of our point generation
        
        # sets all variables to their cleared state
        self.clear_tree()
        
    # Checks if the Tree is valid - it has a valid start, goal, occupancy grid, QuadTree, and world boundaries
    # Returns true if entirely valid, false otherwise
    def valid(self):
        return self.start_valid() and self.goal_valid() and self.occupancy_grid_valid() and self.quad_tree != None and self.world_x_min != None and self.world_x_max != None and self.world_y_min != None and self.world_y_max != None
    
    # Checks we have a starting position (it has a node in the tree)
    # Returns true if the start is valid, false otherwise
    def start_valid(self):
        return len(self.nodes) > 0
    
    # Checks to see if we have a goal position for our travel
    # Returns true if the goal is valid, false otherwise
    def goal_valid(self):
        return self.goal_position != None
    
    # Checks to see if we have an occupancy grid
    # Returns true if the occupancy grid is valid, false otherwise
    def occupancy_grid_valid(self):
        return self.occupancy_grid != None
    
    # Initializes the starting position (root of the Tree)
    def initialize_start(self, start_x, start_y, start_theta): 
        # Updates the world boundaries to include the start position with inflation
        if not self.world_x_min or start_x < self.world_x_min:
            self.world_x_min = start_x - self.world_bounds_inflation
        if not self.world_x_max or start_x > self.world_x_max:
            self.world_x_max = start_x + self.world_bounds_inflation
        if not self.world_y_min or start_y < self.world_y_min:
            self.world_y_min = start_y - self.world_bounds_inflation
        if not self.world_y_max or start_y > self.world_y_max:
            self.world_y_max = start_y + self.world_bounds_inflation
        self.quad_tree = QuadTree(((self.world_x_max + self.world_x_min) / 2.0, (self.world_y_max + self.world_y_min) / 2.0), (self.world_x_max - self.world_x_min), (self.world_y_max - self.world_y_min))    
        
        # Add our starting node to the tree
        self.start_node = Node(RRTPoint(start_x, start_y, start_theta), 0)
        self.add_node(self.start_node)
        
    # Initializes the goal position for the travel
    def initialize_goal(self, goal_x, goal_y, goal_theta = 0):
        # Updates the world boundaries to include the goal position with inflation
        if not self.world_x_min or goal_x < self.world_x_min:
            self.world_x_min = goal_x - self.world_bounds_inflation
        if not self.world_x_max or goal_x > self.world_x_max:
            self.world_x_max = goal_x + self.world_bounds_inflation
        if not self.world_y_min or goal_y < self.world_y_min:
            self.world_y_min = goal_y - self.world_bounds_inflation
        if not self.world_y_max or goal_y > self.world_y_max:
            self.world_y_max = goal_y + self.world_bounds_inflation
        self.quad_tree = QuadTree(((self.world_x_max + self.world_x_min) / 2.0, (self.world_y_max + self.world_y_min) / 2.0), (self.world_x_max - self.world_x_min), (self.world_y_max - self.world_y_min))
        
        # sets the goal position
        self.goal_position = RRTPoint(goal_x, goal_y, goal_theta)
    
    # Initializes the occupancy grid
    def initialize_occupancy_grid(self, occupancy_grid):
        # Updates the world boundaries to include the goal position with inflation
        occupancy_bounds = occupancy_grid.get_occupancy_boundaries()
        if not self.world_x_min or occupancy_bounds[0] < self.world_x_min:
            self.world_x_min = occupancy_bounds[0]
        if not self.world_x_max or occupancy_bounds[1] > self.world_x_max:
            self.world_x_max = occupancy_bounds[1]
        if not self.world_y_min or occupancy_bounds[2] < self.world_y_min:
            self.world_y_min = occupancy_bounds[2]
        if not self.world_y_max or occupancy_bounds[3] > self.world_y_max:
            self.world_y_max = occupancy_bounds[3]
        self.quad_tree = QuadTree(((self.world_x_max + self.world_x_min) / 2.0, (self.world_y_max + self.world_y_min) / 2.0), (self.world_x_max - self.world_x_min), (self.world_y_max - self.world_y_min))

        # sets the occupancy grid
        self.occupancy_grid = occupancy_grid
        
    # Sets the tree to its cleared state (uninitialized tree)
    def clear_tree(self):
        # Represents the boundaries for point generation of our world
        self.world_x_min = None
        self.world_x_max = None
        self.world_y_min = None
        self.world_y_max = None
        
        self.nodes = dict() # A dictionary holding all of our nodes incrementally indexed
        self.quad_tree = None # A QuadTree also holding all the nodes but in a spatially indexed way
        self.occupancy_grid = None # The RRTOccupancyGrid object holding our occupancy grid
        
        self.goal_position = None # The position of our goal for travel
        self.goal_node = None # The Node of our goal (only once the goal gets added to the tree)

        self.max_index = 0 # The maximum index for nodes currently in the tree
     
        
    # Gets the path from the start to the end if it exists
    # Returns the path (list of Nodes) or None if no path exists yet
    def get_path_to_goal(self):
        if self.goal_node is None:
            return None
        else:
            path_list = []
            n = self.goal_node
            while n.parent != None:
                path_list.append(n)
                n = n.parent
            path_list.append(n)

            path_list.reverse()
            return path_list
        
    # Attepts to add the goal node to the tree. 
    # Returns true if successful, returns false if unsuccessful
    def add_goal_node(self):
        goal_node = Node(RRTPoint(self.goal_position.x, self.goal_position.y))
        if self.add_node_to_tree(goal_node):
            self.goal_node = goal_node
            return True
        else:
            return False
        
    # Builds the RRT Tree if there is space with a maximum number of iterations to add a specified number of nodes each time.
    def build_tree(self, nodes_per_iteration, max_iterations):
        iteration = 0
        while not self.add_goal_node() and iteration < max_iterations:
            self.expand_tree(nodes_per_iteration, self.world_x_min, self.world_x_max, self.world_y_min, self.world_y_max)
            self.expand_tree(10, self.start_node.pos.x-10, self.start_node.pos.x+10, self.start_node.pos.y-10, self.start_node.pos.y+10)
                
            # finished one iteration
            iteration += 1
            if iteration == max_iterations:
                return False
        return True
            
    # Adds the specified number of nodes into the tree if there is space with Node generation near the path given.
    def refine_tree(self, num_nodes, path):
        if path is None:
            return

        i = 0
        while not rospy.is_shutdown() and i < num_nodes and len(self.nodes) < self.max_num_nodes:
            path_node = path[random.randint(0, len(path)-1)]
            self.expand_tree(1, path_node.pos.x - self.radius, 
                               path_node.pos.x + self.radius,
                               path_node.pos.y - self.radius,
                               path_node.pos.y + self.radius)
            i += 1
        #x_min, x_max, y_min, y_max = None, None, None, None
        #if path != None:
        #    for n in path:
        #        if not x_min or n.pos.x < x_min:
        #            x_min = n.pos.x - self.world_bounds_inflation
        #        if not x_max or n.pos.x > x_max:
        #            x_max = n.pos.x + self.world_bounds_inflation
        #        if not y_min or n.pos.y < y_min:
        #            y_min = n.pos.y - self.world_bounds_inflation
        #        if not y_max or n.pos.y > y_max:
        #            y_max = n.pos.y + self.world_bounds_inflation
        #            
        #if x_min != None and x_max != None and y_min != None and y_max != None:
        #    self.expand_tree(num_nodes, x_min, x_max, y_min, y_max)

    # Adds the specified number of nodes to the tree if there is space with the given region for Node generation
    def expand_tree(self, num_nodes, x_min, x_max, y_min, y_max):
        i = 0
        while not rospy.is_shutdown() and i < num_nodes and len(self.nodes) < self.max_num_nodes:
            # add random nodes to the tree
            new_node = self.generate_node(x_min, x_max, y_min, y_max)
            if self.node_in_obstacle(new_node):
                continue
            
            self.add_node_to_tree(new_node)

            i += 1

    # Adds the new_node to the tree with re-routing. 
    # Returns true if it was added and false if it was not.
    def add_node_to_tree(self, new_node):
        # get the nearest nodes that are not blocked by an obstacle to the new_node
        nearest_nodes = self.get_nearest_nodes(new_node)
        
        if len(nearest_nodes) == 0:
            # no near nodes with a clear path
            return False
        
        # add a node to our node list
        self.add_node(new_node)
        
        # connect the best possible node to the new node
        best_parent = nearest_nodes[0][0]
        min_cost = best_parent.cost + nearest_nodes[0][1]
        for next_nearest_node in nearest_nodes:
            if next_nearest_node[1] > self.radius:
                continue
            
            n = next_nearest_node[0]
            if n.cost + next_nearest_node[1] < min_cost:
                best_parent = n
                min_cost = best_parent.cost + next_nearest_node[1]
        self.add_connection(best_parent, new_node)

        # go through all the nearest nodes and check re-routing through new_node
        for next_nearest_node in nearest_nodes:
            if next_nearest_node[1] > self.radius:
                continue
            
            n = next_nearest_node[0]
            dist_to_n = next_nearest_node[1]
            self.reassign_parent_if_cheaper(n, new_node)

        return True
    
    # Runs a test to see if the Tree structure is consistent with itself. Checks that the costs of each node are consisten with their parent's cost and checks to see if parent-child pairs are consistent
    def check_consistent(self):
        for index, n in self.nodes.items():
            if n.parent:
                # Parent-Child check
                if n not in n.parent.children:
                    return False
                # Cost check
                if abs(n.parent.cost + n.parent.distance(n) - n.cost) > 0.01:
                    return False
            
        return True

    def update_node_cost(self, node, cost):
        node.cost = cost
        for child in node.children:
            self.update_node_cost(child, cost + node.distance(child))

    def reassign_parent_if_cheaper(self, child, newparent):
        new_cost = newparent.cost + newparent.distance(child)
        if new_cost < child.cost:
            child.parent.children.remove(child)
            child.parent = newparent
            newparent.children.append(child)
            self.update_node_cost(child, new_cost)

    # Makes a connection in the tree from node1 to node2
    def add_connection(self, node1, node2):
        node1.children.append(node2)
        node2.parent = node1
        self.update_node_cost(node2, node1.cost + node1.distance(node2))
        #node2.cost = node1.cost + node1.distance(node2)
        
    # Removes the connection in the tree from node1 to node2
    def remove_connection(self, node1, node2):
        node1.children.remove(node2)
        node2.parent = None
            
    # Adds a node to our data structures (dictionary of nodes and QuadTree)
    def add_node(self, node):
        node.node_index = self.max_index
        self.max_index += 1
        self.nodes[node.node_index] = node
        self.quad_tree.insert((node.pos.x, node.pos.y), data=node)
    
    # Finds the nearest nodes to the given node. NOTE: will either return all nodes within our connection radius, or if there are none then it will return the 10 closest
    # Returns a list of nodes containing [node, distance] sorted from shortest to longest distance.
    def get_nearest_nodes(self, node):
        nearest_nodes = []
        # look within our bounding box with connection radius
        for n in self.get_neighbors_in_box(node.pos.x, node.pos.y):
            n = n.data
            d = sqrt((node.pos.x - n.pos.x) ** 2 + (node.pos.y - n.pos.y) ** 2)
            if d <= self.radius:
                if self.clear_path(n, node): # move this check up
                    nearest_nodes.append([n, d])
            
        
        if False: #if len(nearest_nodes) == 0:
            # if there was nothing in our box look for 10 nearest neighbors
            for n in self.get_nearest_neighbors(node.pos.x, node.pos.y):
                n = n.data
                d = sqrt((node.pos.x - n.pos.x) ** 2 + (node.pos.y - n.pos.y) ** 2)
                if d <= self.radius or len(nearest_nodes) == 0:
                    if self.clear_path(n, node): # move this check up
                        nearest_nodes.append([n, d])
                elif len(nearest_nodes) > 0:
                    break
        
        # Sort based off distance to return closest to farthest
        nearest_nodes = sorted(nearest_nodes, key=lambda x: x[1])
        return nearest_nodes

    # Generates a us a random node. It will sample a random location within the specified boundaries. If the node is larger than connection radius from its next nearest node, the generated node is moved to be within connection radius.
    # Returns the generated Node with the specifications above
    def generate_node(self, x_min, x_max, y_min, y_max):
        x_pos = random.random() * (x_max - x_min) + x_min
        y_pos = random.random() * (y_max - y_min) + y_min
        
        nearest = self.get_nearest_neighbors(x_pos, y_pos)
        if len(nearest) != 0:
            n = self.get_nearest_neighbors(x_pos, y_pos)[0].data
            
            if sqrt((x_pos - n.pos.x) ** 2 + (y_pos - n.pos.y) ** 2) >= self.radius:
                theta = math.atan2((y_pos - n.pos.y), (x_pos - n.pos.x))
                x_pos = n.pos.x + self.radius * math.cos(theta) * (random.random() / 2.0 + 0.5)
                y_pos = n.pos.y + self.radius * math.sin(theta) * (random.random() / 2.0 + 0.5)

        return Node(RRTPoint(x_pos, y_pos))
    
    # Checks if a node is in an obstacle.
    # Returns true if a node is in an obstacle. False otherwise.
    def node_in_obstacle(self, node):
        if self.occupancy_grid is None:
            return False
        
        return self.occupancy_grid.query_obstacle(node.pos.x,node.pos.y)
    
    # Checks if there is a clear straight line path between two nodes
    # Returns true if there is a clear straight line path between two nodes
    def clear_path(self, node1, node2):
        if self.occupancy_grid is None:
            return True
        
        return self.occupancy_grid.query_free(node1.pos.x, node1.pos.y, node2.pos.x, node2.pos.y)
    
    # Get the neighbors within the bounding box specified by connection radius from the given position
    # Returns the generator object for the quad tree based all points in the bounding box
    def get_neighbors_in_box(self, x, y):        
        return self.quad_tree.within_bb(BoundingBox(min_x=x - self.radius, min_y=y - self.radius, max_x=x + self.radius, max_y=y + self.radius))
    
    # Get the 10 nearest neighbors from the specified position
    # Returns the generator object for the quad tree based on the nearest 10 neighbors
    def get_nearest_neighbors(self, x, y):        
        return self.quad_tree.nearest_neighbors((x, y), count=10)

    # Remove a node from the tree (recursively with all children)
    def remove_node(self, node):
        while len(node.children) > 0:
            self.remove_node(node.children[0])
            
        if node.parent:
            self.remove_connection(node.parent, node)

        self.nodes.pop(node.node_index)

    # Resets the quad tree, clearing all nodes then adding back in nodes that are in the dictionary node list.
    def reset_quad(self):
        self.quad_tree = None
        self.quad_tree = QuadTree(((self.world_x_max + self.world_x_min) / 2.0, (self.world_y_max + self.world_y_min) / 2.0), (self.world_x_max - self.world_x_min), (self.world_y_max - self.world_y_min))
        for index, node in self.nodes.items():
            self.quad_tree.insert((node.pos.x, node.pos.y), data=node)

# Class for the interaction with our Tree and the ROS BlueROV system
class RRTPlanner():
    # Initialize with the given parameters
    def __init__(self, connection_radius, max_num_nodes, world_bounds_inflation, num_nodes_refinement, waypoint_tollerance, debug):  
        # PARAMS
        self.debug = debug # Whether or not to run in debug mode (debug gives more information but slows performance)
        self.num_nodes_refinement = num_nodes_refinement # The number of nodes to refine the tree with every cycle
        self.waypoint_tollerance = waypoint_tollerance # The tollerance with which we have reached a Node in our path
        
        self.clear = False # Whether or not the Tree is marked for clearing (will get cleared next cycle to reset goal or occupancy grid)
        self.last_goal = None # The last goal that we recieved
        self.last_occupancy_grid = None # The last occupancy grid that we recieved
        self.robot_position = None # Holds the most recent robot position
        
        # Subscriber to the currently targeted waypoint
        self.target_point_subscriber = rospy.Subscriber(
            '/uuv0/target_waypoint', Point, self.callback_target_waypoint, queue_size=1) 

        # Subscriber to the robots NED position
        self.robot_position_subscriber = rospy.Subscriber(
            '/uuv0/pose_gt_noisy_ned', Odometry, self.callback_robot_position, queue_size=1) 
        
        # Subscriber to the obstacle map
        self.obstacle_map_subscriber = rospy.Subscriber(
            "/uuv0/obstacle_map", OccupancyGrid, self.callback_obstacle_map, queue_size=1)
        
        if self.debug:
            # Publisher for all points in our Tree
            self.rrt_points_pub = rospy.Publisher( '/uuv0/rrt_points', PoseArray, queue_size=1)
            self.rrt_points_msg = PoseArray()
            self.rrt_points_msg.header.frame_id = 'world'
            
        # Path publisher for our planner
        self.rrt_path_pub = rospy.Publisher( '/uuv0/rrt_path', Path, queue_size=1)
        self.rrt_path_msg = Path()
        self.rrt_path_msg.header.frame_id = 'world'

        # Publisher stating whether or not to publish an hsd command (whether or not to move the robot)
        self.publish_rrt_hsd_pub = rospy.Publisher( '/uuv0/publish_rrt_hsd', Bool, queue_size=1)
        
        self.path = None # Holds the path between our starting point and our goal
        self.path_start = None # Holds the starting position of our path (because it gets instantly cleared from the RRT when reached)
        
        self.rrtree = Tree(connection_radius, max_num_nodes, world_bounds_inflation) # Holds our Tree for the RRT

        # RRT controller
        rate = rospy.Rate(15)
        while not rospy.is_shutdown():
            if self.debug:
                print('consistent?',self.rrtree.check_consistent()) # Checks whether or not the RRT is consistent

            # Synchronize the clearing of our trees to prevent None referencing
            if self.clear:
                if self.debug:
                    print('CLEARING')
                self.publish_rrt_hsd_pub.publish(Bool(data = False))
                self.rrtree.clear_tree()
                self.path = None
                self.clear = False
                self.path_start = None
                
            # Initialize the rrt if anything is not valid at the moment
            if self.last_goal != None and not self.rrtree.goal_valid():
                self.rrtree.initialize_goal(self.last_goal[0], self.last_goal[1])
            if self.last_occupancy_grid != None and not self.rrtree.occupancy_grid_valid():
                self.rrtree.initialize_occupancy_grid(self.last_occupancy_grid)
            if self.robot_position != None and self.rrtree.occupancy_grid_valid() and self.rrtree.goal_valid() and not self.rrtree.start_valid():
                self.rrtree.initialize_start(self.robot_position.x, self.robot_position.y, self.robot_position.theta)

            # Build the tree if we are valid and don't have a path
            if self.rrtree.valid() and self.path == None and not self.clear:
                if self.debug:
                    print('BUILDING')
                succ = self.rrtree.build_tree(500, 6) # TODO PARAMETERIZE
                if self.debug:
                    if succ:
                        print('BUILDING COMPLETE')
                    else:
                        print('BUILDING FAILED')
                        self.clear = True

            # Get whatever the current path is
            if self.path == None:
                self.path = self.rrtree.get_path_to_goal()
                if self.path != None:
                    self.extract_first_node_from_tree() # This is our first time getting the path, immediately extract the first node as our path is locked from herea
                    #self.assign_path_start()
            if self.path_start != None:
                self.path = self.rrtree.get_path_to_goal() # Constantly keep getting our path in case it updates downstream

                # Debugging printing the path length
                if self.debug:
                    path_length = 0
                    if self.path != None and len(self.path) >= 2:
                        for i in range(len(self.path) - 1):
                            path_length += self.path[i].distance(self.path[i+1])
                    print('path length:', path_length)

            # If we have reached an RRT waypoint, target the next path waypoint and free up uneeded memory
            while not rospy.is_shutdown() and not self.clear and self.path != None and len(self.path) >= 1 and self.path[0].pos.distance(self.robot_position) < self.waypoint_tollerance:
                self.extract_first_node_from_tree()
                
            # Publish debug topics
            if self.debug:
                self.publish_tree()

            # Publish our path (empty if we don't have one)
            self.publish_path()
            
            # Refine our tree if we have a path
            if self.path != None and len(self.path) >= 1:
                self.rrtree.refine_tree(self.num_nodes_refinement, self.path)
                self.rrtree.expand_tree(self.num_nodes_refinement, self.rrtree.world_x_min, self.rrtree.world_x_max, self.rrtree.world_y_min, self.rrtree.world_y_max)

            # Mark the robot for movement (HSD publishing) if we have a path and are not supposed to be clearing
            if self.path != None and not self.clear:
                self.publish_rrt_hsd_pub.publish(Bool(data = True))
            
            # Sleep
            rate.sleep()

    def assign_path_start(self):
        self.path_start = self.path[0]
        self.path = self.rrtree.get_path_to_goal()


    # Change our path start to whatever the current first node is in the path and remove that node from from the rrtree.
    # Essentially moves our path waypoint forward by one and clears up any unneeded memory from the tree
    def extract_first_node_from_tree(self):
        temp = self.path[0]
        if len(self.path) >= 2:
            self.rrtree.remove_connection(self.path[0], self.path[1])
        else:
            self.rrtree.goal_node = None
        self.rrtree.remove_node(self.path[0])
        self.rrtree.reset_quad()
        self.path = self.rrtree.get_path_to_goal()
        self.path_start = temp

    # Publishes all the current nodes in the tree
    def publish_tree(self):
        self.rrt_points_msg.poses = []
        self.rrt_points_msg.header.seq += 1
        self.rrt_points_msg.header.stamp.secs += 1
        if self.rrtree.nodes != None:
            for index, node in self.rrtree.nodes.items():
                self.rrt_points_msg.poses.append(Pose())
                self.rrt_points_msg.poses[-1].position.x = node.pos.x
                self.rrt_points_msg.poses[-1].position.y = node.pos.y
                self.rrt_points_msg.poses[-1].position.z = -45.0
                Q = Rotation.from_euler('XYZ', [0,0,node.pos.theta]).as_quat()
                self.rrt_points_msg.poses[-1].orientation.x = Q[0]
                self.rrt_points_msg.poses[-1].orientation.y = Q[1]
                self.rrt_points_msg.poses[-1].orientation.z = Q[2]
                self.rrt_points_msg.poses[-1].orientation.w = Q[3]
        self.rrt_points_pub.publish(self.rrt_points_msg)
        
    # Publishes our path if we have one, otherwise publishes an empty path
    def publish_path(self):
        self.rrt_path_msg.poses = []
        self.rrt_path_msg.header.seq += 1
        self.rrt_path_msg.header.stamp.secs += 1
        if self.path_start != None and self.path != None and len(self.path) >= 1:
            pos = self.path_start.pos
            self.rrt_path_msg.poses.append(PoseStamped())
            self.rrt_path_msg.poses[-1].header.seq = self.rrt_path_msg.header.seq
            self.rrt_path_msg.poses[-1].header.stamp = self.rrt_path_msg.header.stamp
            self.rrt_path_msg.poses[-1].header.frame_id = self.rrt_path_msg.header.frame_id
            self.rrt_path_msg.poses[-1].pose.position.x = pos.x
            self.rrt_path_msg.poses[-1].pose.position.y = pos.y
            self.rrt_path_msg.poses[-1].pose.position.z = -45.0
            Q = Rotation.from_euler('XYZ', [0,0,pos.theta]).as_quat()
            self.rrt_path_msg.poses[-1].pose.orientation.x = Q[0]
            self.rrt_path_msg.poses[-1].pose.orientation.y = Q[1]
            self.rrt_path_msg.poses[-1].pose.orientation.z = Q[2]
            self.rrt_path_msg.poses[-1].pose.orientation.w = Q[3]
            
            for node in self.path:
                pos = node.pos
                self.rrt_path_msg.poses.append(PoseStamped())
                self.rrt_path_msg.poses[-1].header.seq = self.rrt_path_msg.header.seq
                self.rrt_path_msg.poses[-1].header.stamp = self.rrt_path_msg.header.stamp
                self.rrt_path_msg.poses[-1].header.frame_id = self.rrt_path_msg.header.frame_id
                self.rrt_path_msg.poses[-1].pose.position.x = pos.x
                self.rrt_path_msg.poses[-1].pose.position.y = pos.y
                self.rrt_path_msg.poses[-1].pose.position.z = -45.0
                Q = Rotation.from_euler('XYZ', [0,0,pos.theta]).as_quat()
                self.rrt_path_msg.poses[-1].pose.orientation.x = Q[0]
                self.rrt_path_msg.poses[-1].pose.orientation.y = Q[1]
                self.rrt_path_msg.poses[-1].pose.orientation.z = Q[2]
                self.rrt_path_msg.poses[-1].pose.orientation.w = Q[3]
        
        self.rrt_path_pub.publish(self.rrt_path_msg)
    
    # Gets a new obstacle map from ROS. If the new map intersects our path, clear rrtree and don't move the robot!
    def callback_obstacle_map(self, msg):
        self.last_occupancy_grid = RRTOccupancyGrid(msg)
        path = copy.deepcopy(self.path)
        if path != None and self.path_start != None:
            path.insert(0, self.path_start)
        if self.last_occupancy_grid != None and not self.last_occupancy_grid.query_path_free(path): # recomputation is necessary
            self.clear = True
            self.publish_rrt_hsd_pub.publish(Bool(data = False))

    # Gets a new robot position from ROS.
    def callback_robot_position(self, msg):
        # Change our robot position to ENU
        x, y, theta = NED_to_ENU(msg.pose.pose.position.x, msg.pose.pose.position.y, get_yaw_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]))
        self.robot_position = RRTPoint(x, y, theta)

    # Gets a new target waypoint from ROS. If the waypoint is new, clear the rrtree and don't move the robot.
    def callback_target_waypoint(self, msg):
        x = msg.x
        y = msg.y
        
        if self.rrtree.goal_valid() and (abs(x - self.rrtree.goal_position.x) > 0.01 or abs(y - self.rrtree.goal_position.y) > 0.01):
            # we have a new waypoint
            self.clear = True
            self.publish_rrt_hsd_pub.publish(Bool(data = False))
        self.last_goal = (x,y)

# Main entry point of the script. Setup the RRTPlanner with parameters
if __name__=='__main__':
    rospy.init_node('rrt_planner', log_level=rospy.INFO)
    try:
        rrt_planner = RRTPlanner(25.0, 3000, 75, 25, 4, True)
        rospy.spin()
    except rospy.ROSInterruptException as e:
        print('caught exception:', e)
