import numpy as np


class Gridworld:

    def __init__(self):
        self.V = np.zeros([16, 1])

        #  Define Rewards table
        #  [ 0,  1,  2,  3]
        #  [ 4,  5,  6,  7]
        #  [ 8,  9, 10, 11]
        #  [12, 13, 14, 15]
        self.R = np.asarray([0,  0, 0, 100,
                             0,-10, 0, -10,
                             0,  0, 20,-10,
                             0,-10,-10,-10])

        self.non_terminal_states = [2,1,0,6,4,12,10,9,8]
        #  action order: up, right, down, left.
        #  up = 0. right = 1. down = 2. left = 3.
        #  States ordered from top right, descending: [[3,2,1,0],...]

    def initialize(self):
        """
        Initializes an empty value table and policy table
        """
        V = np.zeros([len(self.R), 1])
        policy = np.ones([len(self.R), 4]) * 0.25
        return V, policy


    def transition_function(self, state, action):
        state_probabilities = [[1,state]]
        if state == 0:
            if action == 1:
                state_probabilities = [[0.95,1],[0.05,3]]
            if action == 2:
                state_probabilities = [[0.95,4],[0.05,12]]
        elif state == 1:
            if action == 1:
                state_probabilities = [[0.95,2],[0.05,3]]
            if action == 2:
                state_probabilities = [[1,5]]
            if action == 3:
                state_probabilities = [[1,0]]
        elif state == 2:
            if action == 1:
                state_probabilities = [[1,3]]
            if action == 2:
                state_probabilities = [[0.95,6],[0.05,14]]
            if action == 3:
                state_probabilities = [[0.95,1],[0.05,0]]
        elif state == 4:
            if action == 0:
                state_probabilities = [[1,0]]
            if action == 1:
                state_probabilities = [[1,5]]
            if action == 2:
                state_probabilities = [[0.95,8],[0.05,12]]
        elif state == 6:
            if action == 0:
                state_probabilities = [[1,2]]
            if action == 1:
                state_probabilities = [[1,7]]
            if action == 2:
                state_probabilities = [[0.95,10],[0.05,14]]
            if action == 3:
                state_probabilities = [[1,5]]
        elif state == 8:
            if action == 0:
                state_probabilities = [[0.95,4],[0.05,0]]
            if action == 1:
                state_probabilities = [[0.95,9],[0.05,11]]
            if action == 2:
                state_probabilities = [[1,12]]
        elif state == 9:
            if action == 0:
                state_probabilities = [[1,5]]
            if action == 1:
                state_probabilities = [[0.95,10],[0.05,11]]
            if action == 2:
                state_probabilities = [[1,13]]
            if action == 3:
                state_probabilities = [[1,8]]
        elif state == 10:
            if action == 0:
                state_probabilities = [[0.95,6],[0.05,2]]
            if action == 1:
                state_probabilities = [[1,11]]
            if action == 2:
                state_probabilities = [[1,14]]
            if action == 3:
                state_probabilities = [[0.95,9],[0.05,8]]
        elif state == 12:
            if action == 0:
                state_probabilities = [[0.95,8],[0.05,0]]
            if action == 1:
                state_probabilities = [[1,13]]
        return state_probabilities
