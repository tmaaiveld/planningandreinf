import numpy as np
import time
import random

#Initial Q-space, #Initial V-space
# Q1 = np.zeros([16, 4])
V1 = np.zeros([16, 1])

#Updated Q-space, #Updated V-space
# Q2 = np.zeros([16, 4])

#Define Rewards table
#[ 0,  1,  2,  3]
#[ 4,  5,  6,  7]
#[ 8,  9, 10, 11]
#[12, 13, 14, 15]
R1 = np.asarray([0, 0, 0, 100, 
                 0, -10, 0, -10, 
                 0, 0, 20, -10, 
                 0, -10, -10, -10])

# Initialize a random policy
policy = np.ones([16, 4]) * 0.25 # terminal states won't be picked anyway

#Keep track of amount of loops, for discount factor
amount_of_steps = 0

non_terminal_states = [0,1,2,4,6,8,9,10,12]
terminal_states = [3,5,7,11,13,14,15]
#action order: up, right, down, left.
#up = 0. right = 1. down = 2. left = 3.

#Discount
THETA = 0.00001

def return_stateprobabilities(state,action):
	state_probabilities = [[1,state]]
	if state == 0:
		if action == 1:
			state_probabilities =  [[0.95,1],[0.05,3]]
		if action == 2:
			state_probabilities =  [[0.95,4],[0.05,12]]
	elif state == 1:
		if action == 1:
			state_probabilities =  [[0.95,2],[0.05,3]]
		if action == 2:
			state_probabilities =  [[1,5]]
		if action == 3:
			state_probabilities =  [[1,0]]
	elif state == 2:
		if action == 1:
			state_probabilities =  [[1,3]]
		if action == 2:
			state_probabilities =  [[0.95,6],[0.05,14]]
		if action == 3:
			state_probabilities =  [[0.95,1],[0.05,0]]
	elif state == 4:
		if action == 0:
			state_probabilities = [[1,0]]
		if action == 1:
			state_probabilities =  [[1,5]]
		if action == 2:
			state_probabilities =  [[0.95,8],[0.05,12]]
	elif state == 6:
		if action == 0:
			state_probabilities =  [[1,2]]
		if action == 1:
			state_probabilities =  [[1,7]]
		if action == 2:
			state_probabilities =  [[0.95,10],[0.05,14]]
		if action == 3:
			state_probabilities =  [[1,5]]
	elif state == 8:
		if action == 0:
			state_probabilities =  [[0.95,4],[0.05,0]]
		if action == 1:
			state_probabilities =  [[0.95,9],[0.05,11]]
		if action == 2:
			state_probabilities =  [[1,12]]
	elif state == 9:
		if action == 0:
			state_probabilities =  [[1,5]]
		if action == 1:
			state_probabilities =  [[0.95,10],[0.05,11]]
		if action == 2:
			state_probabilities =  [[1,13]]
		if action == 3:
			state_probabilities =  [[1,8]]
	elif state == 10:
		if action == 0:
			state_probabilities =  [[0.95,6],[0.05,2]]
		if action == 1:
			state_probabilities =  [[1,11]]
		if action == 2:
			state_probabilities =  [[1,14]]
		if action == 3:
			state_probabilities =  [[0.95,9],[0.05,8]]
	elif state == 12:
		if action == 0:
			state_probabilities =  [[0.95,8],[0.05,0]]
		if action == 1:
			state_probabilities =  [[1,13]]
	return state_probabilities


def solution_set(policy):
    solution_matrix = np.chararray([16])
    p = 0
    for row in policy:
        if row[0] == 0.25:
            solution_matrix[p] = "T"
        elif row[0] == 1:
            solution_matrix[p] = "U"
        elif row[1] == 1:
            solution_matrix[p] = "R"
        elif row[2] == 1:
            solution_matrix[p] = "D"
        elif row[3] == 1:
            solution_matrix[p] = "L"
        p+=1
    return np.reshape(solution_matrix, [4,4])

i = 0

while True:
    evaluation_counter = 0
    while True: # step 2
        delta = 0
        for state in non_terminal_states:
            v = np.copy(V1[state])
            V1[state] = 0
            for action in range(4):
                for possible_outcome in return_stateprobabilities(state,action):
                    trans_prob = possible_outcome[0]              
                    next_state = possible_outcome[1]

                    V1[state] += policy[state,action] * trans_prob * (R1[next_state] + GAMMA * V1[next_state])

            delta = max(delta, abs(v - V1[state]))

        evaluation_counter += 1
        print('I have completed {} evaluation loop(s).'.format(evaluation_counter))

        if delta < THETA:
            break
        

# step 3
    policy_stable = True
    for state in non_terminal_states:
        old_policy = np.copy(policy[state])
        action_values = np.zeros([4,1])
        for action in range(4):
            for possible_outcome in return_stateprobabilities(state,action):
                trans_prob = possible_outcome[0]              
                next_state = possible_outcome[1] 
                action_values[action] += trans_prob * (R1[next_state] + GAMMA * V1[next_state])

        policy[state, np.argmax(action_values)] = 1
        policy[state, np.arange(4)!=np.argmax(action_values)] = 0
        
        # print("\n old policy: {}".format(old_policy))
        # print("\n new: {}".format(policy[state]))

        if not np.array_equal(old_policy, policy[state]):
            policy_stable = False

    i+=1
    print("I've completed {} improvement loop(s).".format(i))

    if policy_stable == True:
        print("\n final policy")
        print(solution_set(policy))

        print("\nValue table")
        print(np.reshape(V1, [4,4]))
        break