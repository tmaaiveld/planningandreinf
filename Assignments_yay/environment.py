import numpy as np
import random

'''Might write a rewards counter or perhaps in algo module itself'''
class Gridworld:

    def __init__(self):
        self.Q = np.zeros([16, 4])

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
 

    def initialize(self, epsilon):
        """
        Initializes an empty value table and policy table
        """
        Q = np.zeros([len(self.R), 4])
        V = np.zeros([len(self.R), 1])
        policy = np.ones([len(self.R), 4]) * (epsilon/4)
        return Q, V, policy #, V, policy


    def initialize_state(self):
        initial_state = random.choice(self.non_terminal_states)
        return initial_state


    def e_greedy_action_selection(self, Q, state, epsilon):
        """
        Chooses an action for a given state according to an epsilon-greedy policy
        """
        policy[state, np.argmax(Q[state])] = 1 - epsilon

        chosen_action = np.random.choice(4, 1, p = policy[state]) 
        return chosen_action

    
    def take(self, state, action):
        '''
        Take some action from some state and observe the reward and next state
        '''
        #  for possible_outcome in transition_function(state, action):  # not sure if this will work
		# 		trans_prob = possible_outcome[0]
		# 		next_state = possible_outcome[1]
        # ### choose randomly if there's multiple outcomes
        next_state = transition_function(state, action)[1]

        reward = self.R[next_state]
    
        return reward, next_state
        

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
