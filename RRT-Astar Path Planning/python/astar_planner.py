#!/usr/bin/python
import sys
import time
import pickle
import numpy as np
from itertools import product
from math import cos, sin, pi, sqrt 

from plotting_utils import draw_plan
from priority_queue import priority_dict

from datetime import datetime # time the performance between ASTAR and DIJSTRA

class State(object):
    """
    2D state. 
    """
    
    def __init__(self, x, y):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        # want to make sure that the state assigned does not exceed the max/ min of the map
        if (x>650) or (y>650):
            raise Exception('Coordinates should not exceed (650,650). The value of x,y was: {},{}'.format(x,y))
        elif (x<0) or (y<0):
            raise Exception('Coordinates should not be less than (0,0). The value of x,y was: {},{}'.format(x,y))
        else:
            self.x = x
            self.y = y

        
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
    
    
class AStarPlanner(object):
    """
    Applies the A* shortest path algorithm on a given grid world
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
        
    def get_neighboring_states(self, state):
        """
        Returns free neighboring states of the given state. Returns up to 8
        neighbors (north, south, east, west, northwest, northeast, southwest, southeast)
        """
        
        x = state.x
        y = state.y
        
        # Neighbors can only lie within the image. This is how to get the image size.
        rows, cols = self.world.shape[:2]

        dx = [0]
        dy = [0]
        
        if (x > 0):
            dx.append(-1)

        if (x < rows -1):
            dx.append(1)

        if (y > 0):
            dy.append(-1)

        if (y < cols -1):
            dy.append(1)

        # product() returns the cartesian product
        # yield is a python generator. Look it up.
        for delta_x, delta_y in product(dx,dy):
            if delta_x != 0 or delta_y != 0:
                ns = State(x + delta_x, y + delta_y)
                if self.state_is_free(ns):
                    yield ns 
            

    def _follow_parent_pointers(self, parents, state):
        """
        Assumes parents is a dictionary. parents[key]=value
        means that value is the parent of key. If value is None
        then key is the starting state. Returns the shortest
        path [start_state, ..., destination_state] by following the
        parent pointers.
        """
        
        assert (state in parents)
        curr_ptr = state
        shortest_path = [state]
        
        while curr_ptr is not None:
            shortest_path.append(curr_ptr)
            curr_ptr = parents[curr_ptr]

        # return a reverse copy of the path (so that first state is starting state)
        return shortest_path[::-1]

    #
    # Method accepts additional type parameter [ASTAR,DIJKSTRA]
    #
    def plan(self, start_state, dest_state, model_type):
        """
        Returns the shortest path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # Q is a mutable priority queue implemented as a dictionary
        Q = priority_dict()
        Q[start_state] = 0.0

        # Array that contains the optimal distance we've found from the starting state so far
        best_dist_found = float("inf") * np.ones((world.shape[0], world.shape[1]))
        best_dist_found[start_state.x, start_state.y] = 0

        # Boolean array that is true iff the distance to come of a state has been
        # finalized
        visited = np.zeros((world.shape[0], world.shape[1]), dtype='uint8')

        # Contains key-value pairs of states where key is the parent of the value
        # in the computation of the shortest path
        parents = {start_state: None}
        
        while Q:
            
            # s is also removed from the priority Q with this
            s = Q.pop_smallest()

            # Assert s hasn't been visited before
            assert (visited[s.x, s.y] == 0)

            # Mark it visited because here we will go over every neighbor, 
            # so there's no need to come back after this (by greedy property)
            visited[s.x, s.y] = 1
            
            if s == dest_state:
                return self._follow_parent_pointers(parents, s)

            # for all free neighboring states
            for ns in self.get_neighboring_states(s):
                if visited[ns.x, ns.y] == 1:
                    continue

                transition_distance = sqrt((ns.x - s.x)**2 + (ns.y - s.y)**2)
                
                if(model_type=="ASTAR"):
                    astar_distance = sqrt((dest_state.x-ns.x)**2 + (dest_state.y-ns.y)**2)
                    alternative_best_dist_ns = best_dist_found[s.x, s.y] + transition_distance + astar_distance
                elif(model_type=="DIJKSTRA"):
                    alternative_best_dist_ns = best_dist_found[s.x, s.y] + transition_distance
                # print("transition_distance: ", transition_distance)
                # print("astar_distance: ", astar_distance)
                # print("total: ",alternative_best_dist_ns)
                # if the state ns has not been visited before or we just found a shorter path
                # to visit it then update its priority in the queue, and also its
                # distance to come and its parent
                if (ns not in Q) or (alternative_best_dist_ns < best_dist_found[ns.x, ns.y]):
                    Q[ns] = alternative_best_dist_ns
                    best_dist_found[ns.x, ns.y] = alternative_best_dist_ns
                    parents[ns] = s
                    
        return [start_state]

"""
Helper method to run the algorithms 3 times based on different starting/destination points
the pictures are stored in ../plan folder

"""
def run_helper(world,planning_model,start_state,dest_state,model_type,filepath):
    start=datetime.now()
    try:
        start_state = start_state
        dest_state = dest_state
        plan = astar.plan(start_state, dest_state,model_type="ASTAR")
        draw_plan(
            world = world, 
            plan = plan,
            filepath = filepath     
        )
    except Exception as e:
        print(e)
    print("Model runtime is {} for model {} ".format(datetime.now()-start,model_type))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: astar_planner.py occupancy_grid.pkl"
        sys.exit(1)

    pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()

    astar = AStarPlanner(world)
    
    """
    Below is to bench mark the performance of each of the methods and generate images
    the pictures are stored in ../plan folder
    """
    run_helper(
        world = world,
        planning_model = astar,
        start_state = State(10, 10),
        dest_state = State(630, 600) ,
        model_type = "ASTAR",
        filepath = "../plans/astar_result0_jamestang.png"
    )

    # run_helper(
    #     world = world,
    #     planning_model = astar,
    #     start_state = State(10, 10),
    #     dest_state = State(630, 600),
    #     model_type = "DIJKSTRA",
    #     filepath = "../plans/dijstra_result0_jamestang.png"
    # )

    run_helper(
        world = world,
        planning_model = astar,
        start_state = State(45, 30),
        dest_state = State(100, 640) ,
        model_type = "ASTAR",
        filepath = "../plans/astar_result1_jamestang.png"
    )

    # run_helper(
    #     world = world,
    #     planning_model = astar,
    #     start_state = State(45, 30),
    #     dest_state = State(300, 640),
    #     model_type = "DIJKSTRA",
    #     filepath = "../plans/dijstra_result1_jamestang.png"
    # )


    run_helper(
        world = world,
        planning_model = astar,
        start_state = State(15, 110),
        dest_state = State(400, 340) ,
        model_type = "ASTAR",
        filepath = "../plans/astar_result2_jamestang.png"
    )

    # run_helper(
    #     world = world,
    #     planning_model = astar,
    #     start_state = State(15, 110),
    #     dest_state = State(400, 340) ,
    #     model_type = "DIJKSTRA",
    #     filepath = "../plans/dijstra_result2_jamestang.png"
    # )
    
