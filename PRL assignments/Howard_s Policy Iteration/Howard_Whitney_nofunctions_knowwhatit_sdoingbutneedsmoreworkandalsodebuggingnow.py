import numpy as np
import time

#Initial Q-space
Q1 = np.zeros([16, 4])

#Updated Q-space
Q2 = np.zeros([16, 4])

#Define Rewards table
#[ 0,  1,  2,  3]
#[ 4,  5,  6,  7]
#[ 8,  9, 10, 11]
#[12, 13, 14, 15]
R1 = np.asarray([[0,   0,   0, 100,
				  0, -10,   0, -10,
				  0,   0,  20, -10,
				  0, -10, -10, -10]])

#Keep track of amount of loops, for discount factor
amount_of_steps = 0

non_terminal_states = [0,1,2,4,6,8,9,10,12]
#action order: up, right, down, left.
#up = 0. right = 1. down = 2. left = 3.

#Discount
GAMMA = 0.9


def return_stateprobabilities(state,action):
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


### ___________ POLICY EVALUATION _______________ ###
action_prob = 0.25
improveable = np.zeros([16,1])

# make states non-terminal
for state in non_terminal_states:
	improveable[state] = 1

while True:
	amount_of_steps +=1 
	#Update our known data
	Q1 = np.copy(Q2)
	# improveable = np.zeros([16,1])
	# for state in non_terminal_states:
	# 	improveable[state] = 1
	policy = []

	for state in non_terminal_states:
			if improveable[state] == 1:
				state_value_update = 0
				for action in range(4):
					trans_prob = return_stateprobabilities(state, action)
					term = action_prob * trans_prob[0][0] * (R1[0,trans_prob[0][1]] + GAMMA * (np.max(Q1[trans_prob[0][1]])))
					state_value_update += term    
					print('Updated state value of state ' + str(state) + ' is ' + str(state_value_update))
					Q2[state, action] = state_value_update

############################ Improvement #################################	
	for state in non_terminal_states:
        #state_value_update = 0
		action_values = []
		for action in range(4):
			trans_prob = return_stateprobabilities(state,action)
			action_value = trans_prob[0][0] * (R1[0,trans_prob[0][1]] + GAMMA * (np.max(Q1[trans_prob[0][1]])))
			action_values.append(action_value)
		
		if max(action_values) <= np.amax(Q2, axis=1)[state]:
			improveable[state] = 0
					
	# This if statement checks for converging, and prints tables.
	if np.array_equal(Q1,Q2):
		print("Completed {} cycle(s)".format(amount_of_steps))
		print("The tables are identical, the policy has converged.")
		print(Q1)
		final_Q_values = np.amax(Q1, axis=1)
		print("")
		print("The values of all 16 states are shown below")
		print(final_Q_values[0:4])
		print(final_Q_values[4:8])
		print(final_Q_values[8:12])
		print(final_Q_values[12:16])
		break
		
	# This allows us to keep track of how many loops we've done
	# time.sleep(1)
	print("Completed {} cycle(s)".format(amount_of_steps))