import numpy as np
import random

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
 

    def initialize(self):    # Might rewrite this, don't need V and Policy really
        """
        Initializes an empty value table and policy table
        """
        Q = np.zeros([len(self.R), 4])
        V = np.zeros([len(self.R), 1])
        policy = np.ones([len(self.R), 4]) * 0.25
        return Q, V, policy 


    def initialize_state(self):
        initial_state = random.choice(self.non_terminal_states)
        return initial_state


    def initialize_action(self):
        initial_action = random.randint(0,3)
        return initial_action


    def e_greedy_action_selection(self, Q, state, epsilon):   
        """
        Chooses an action for a given state according to an epsilon-greedy policy      
        """
        best_action = np.argmax(Q[state]) if not all(action == Q[state][0] for action in Q[state]) else random.randint(0,3)
        chosen_action = self.explore(best_action) if random.uniform(0,1) < epsilon else best_action
        return chosen_action


    def explore(self, greedy_action):
        actions = list(range(4))
        actions.remove(greedy_action)
        exploratory_action = random.choices(actions)
        return exploratory_action[0]


    def softmax_action_selection(self, Q, state, temperature):
        probabilities = self.softmax(Q[state], temperature)
        # print(probabilities)
        chosen_action = self.weighted_choice(probabilities)
        return chosen_action


    def softmax(self, x, T):
        """Transform list contents to softmax values."""
        # print(x/T)
        soft_x = np.exp(x/T) / np.sum(np.exp(x/T), axis=0)
        # print(soft_x)
        return soft_x


    def weighted_choice(self, weights):
        totals = np.cumsum(weights)
        throw = np.random.rand()
        return np.searchsorted(totals, throw)


    def take(self, state, action):
        '''
        Take some action from some state and observe the reward and next state 
        '''
        if type(state) == list:
            state = state[0]

        possible_outcome = self.transition_function(state, action)
        next_state = random.choices([possible_outcome[0][1],possible_outcome[-1][1]], [possible_outcome[0][0],possible_outcome[-1][0]], k=1)

        reward = self.R[next_state]

        if type(next_state) == list:
            next_state = next_state[0]
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
