#!/usr/bin/python
import sys
import time
import pickle
import numpy as np
import random
import cv2

from itertools import product
from math import cos, sin, pi, sqrt, atan, floor, degrees

from plotting_utils import draw_plan
from priority_queue import priority_dict

class State(object):
    """
    2D state. 
    """
    
    def __init__(self, x, y, parent):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []

        
    def __eq__(self, state):
        """
        When are two states equal?
        """    
        return state and self.x == state.x and self.y == state.y 

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash((self.x, self.y))
    
    def euclidean_distance(self, state):
        assert (state)
        return sqrt((state.x - self.x)**2 + (state.y - self.y)**2)
    
class RRTPlanner(object):
    """
    Applies the RRT algorithm on a given grid world
    """
    
    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')
        
    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby 
        surroundings are free.
        """
        return (self.occ_grid[state.y-5:state.y+5, state.x-5:state.x+5] == 0).all()

    def is_interpolated_state_free(self,state,inteprolated_distance):
        return (self.occ_grid[state.y-inteprolated_distance:state.y+inteprolated_distance, state.x-inteprolated_distance:state.x+inteprolated_distance] == 0).all()

    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image. 
        """
        x = np.random.random_integers(0, 649)
        y = np.random.random_integers(0, 649)
        return State(x, y, None)
           

    def _follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """
        
        curr_ptr = state
        path = [state]
        
        while curr_ptr is not None:
            path.append(curr_ptr)
            curr_ptr = curr_ptr.parent

        # return a reverse copy of the path (so that first state is starting state)
        return path[::-1]


    def find_closest_state(self, tree_nodes, state):
        min_dist = float("Inf")
        closest_state = None
        for node in tree_nodes:
            dist = node.euclidean_distance(state)  
            if dist < min_dist:
                closest_state = node
                min_dist = dist

        return closest_state

    def steer_towards(self, s_nearest, s_rand, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:
        
        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y
        
        Otherwise, s_rand is farther than max_radius from s_nearest. 
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.
        
        """
        
        if(sqrt((s_rand.x-s_nearest.x)**2 + (s_rand.y-s_nearest.y)**2) < max_radius):
            x = s_rand.x
            y = s_rand.y
        else:
            if (s_rand.x-s_nearest.x) == 0:
                angle_in_rad = atan(0)
            else:
                angle_in_rad = atan((float(s_rand.y)-s_nearest.y)/(s_rand.x-s_nearest.x))
            # print(s_rand.x,s_rand.y)
            # print(s_nearest.x,s_nearest.y)
            # print(degrees(angle_in_rad))
            x = int(floor(max_radius * cos(angle_in_rad))) + s_nearest.x
            y = int(floor(max_radius * sin(angle_in_rad))) + s_nearest.y

        s_new = State(x, y, s_nearest)
        return s_new



    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        # print("s_from x | y : {} | {}".format(s_from.x,s_from.y))
        # print("s_to x | y : {} | {}".format(s_to.x,s_to.y))

        assert (self.state_is_free(s_from))
        
        if not (self.state_is_free(s_to)):
            return False

        max_checks = 10
        if ((s_to.x-s_from.x) == 0):
            for i in xrange(max_checks):
                dist_from_line = float(i)/max_checks * sqrt((s_from.x-s_to.x)**2 + (s_from.y-s_to.y)**2)
                for increment in range(s_to.y-s_from.y):
                    y_pos = increment + s_from.y
                    x_pos_1 = s_to.x + dist_from_line
                    x_pos_2 = s_to.x - dist_from_line
                    if ( not self.state_is_free(State(int(x_pos_1),int(y_pos),None))) and ( not self.state_is_free(State(int(x_pos_2),int(y_pos),None))):
                        return False          
            # Otherwise the line is free, so return true
            return True
        elif((s_to.y-s_from.y) == 0):
            for i in xrange(max_checks):
                dist_from_line = float(i)/max_checks * sqrt((s_from.x-s_to.x)**2 + (s_from.y-s_to.y)**2)
                for increment in range(s_to.x-s_from.x):
                    x_pos = increment + s_from.x
                    y_pos_1 = s_to.y + dist_from_line
                    y_pos_2 = s_to.y - dist_from_line
                    if ( not self.state_is_free(State(int(x_pos),int(y_pos_1),None))) and ( not self.state_is_free(State(int(x_pos),int(y_pos_2),None))):
                        return False       
            # Otherwise the line is free, so return true
            return True
        else:
            slope_m = float(s_to.y-s_from.y)/(s_to.x-s_from.x)
            angle_in_rad = atan(slope_m)
            slope_b = s_to.y - s_to.x * slope_m
            # print("M : {} | B : {}".format(slope_m,slope_b))
            for i in xrange(max_checks):
                dist_from_line = float(i)/max_checks * sqrt((s_from.x-s_to.x)**2 + (s_from.y-s_to.y)**2)
                for increment in range(s_to.x-s_from.x):
                    x_pos = s_from.x + increment # x postion on line
                    x_pos_1 = s_from.x + increment + dist_from_line * cos(angle_in_rad) # x positive interpolation
                    x_pos_2 = s_from.x + increment - dist_from_line * cos(angle_in_rad) # x negative interpolation
                    y_pos_1 = x_pos * slope_m + slope_b + dist_from_line * sin(angle_in_rad) # x positive interpolation
                    y_pos_2 = x_pos * slope_m + slope_b - dist_from_line * sin(angle_in_rad) # y negative interpolation
                    if ( not self.state_is_free(State(int(x_pos_1),int(y_pos_2),None))) and ( not self.state_is_free(State(int(x_pos_2),int(y_pos_1),None))):
                        return False   
            # Otherwise the line is free, so return true
            return True
    
    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius,filepath):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # The set containing the nodes of the tree
        tree_nodes = set()
        tree_nodes.add(start_state)
        
        # image to be used to display the tree
        img = np.copy(self.world)

        plan = [start_state]
        
        cv2.circle(img, center = (start_state.x,start_state.y), radius= 3, color= (0,0,255))
        cv2.circle(img, center = (dest_state.x,dest_state.y), radius = 3, color= (0,255,0))
        for step in xrange(max_num_steps):

            s_rand = self.sample_state()
            s_nearest = self.find_closest_state(tree_nodes=tree_nodes, state=s_rand)
            s_new =  self.steer_towards(s_nearest, s_rand, max_steering_radius)
            
            if self.path_is_obstacle_free(s_nearest, s_new):
                tree_nodes.add(s_new)
                s_nearest.children.append(s_new)

                # If we approach the destination within a few pixels
                # we're done. Return the path.
                if s_new.euclidean_distance(dest_state) < dest_reached_radius:
                    dest_state.parent = s_new
                    plan = self._follow_parent_pointers(dest_state)
                    break
                
                # plot the new node and edge
                cv2.circle(img, (s_new.x, s_new.y), 2, (0,0,0))
                cv2.line(img, (s_nearest.x, s_nearest.y), (s_new.x, s_new.y), (255,0,0))

            # Keep showing the image for a bit even
            # if we don't add a new node and edge
            cv2.imshow('image', img)
            cv2.waitKey(10)

        draw_plan(img, plan,filepath, bgr=(0,0,255), thickness=2)
        cv2.waitKey(0)
        return [start_state]


    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: rrt_planner.py occupancy_grid.pkl"
        sys.exit(1)

    pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()

    rrt = RRTPlanner(world)

    start_state = State(10, 10, None)
    dest_state = State(550, 500, None)

    max_num_steps = 1000     # max number of nodes to be added to the tree 
    max_steering_radius = 30 # pixels
    dest_reached_radius = 50 # pixels
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius,
                    filepath = "../plans/rrt_result2_jamestang.png"
                    )