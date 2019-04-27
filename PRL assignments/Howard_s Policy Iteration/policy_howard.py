import numpy as np
import time
import random

#Initial Q-space
Q1 = np.zeros([16, 4])

#Updated Q-space
Q2 = np.zeros([16, 4])

#Define Rewards table
#[ 0,  1,  2,  3]
#[ 4,  5,  6,  7]
#[ 8,  9, 10, 11]
#[12, 13, 14, 15]
R1 = np.asarray([[0, 0, 0, 100, 0, -10, 0, -10, 0, 0, 20, -10, 0, -10, -10, -10]])

#Keep track of amount of loops, for discount factor
amount_of_steps = 0

non_terminal_states = [0,1,2,4,6,8,9,10,12]
terminal_states = [3,5,7,11,13,14,15]
#action order: up, right, down, left.
#up = 0. right = 1. down = 2. left = 3.

#Discount
GAMMA = 0.9


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
	

while True:
	#Update our known data
	Q1 = np.copy(Q2) 
	amount_of_steps += 1
	optimal_policy = np.argmax(Q1, axis=1)
	for i in non_terminal_states:

		#Action probabilities for each action, will be 1 when added up
		action_prob = [0, 0, 0, 0]			
		optimal_action = optimal_policy[i]

		#If there is an optimal policy, we do that with probability 1. The rest is 0.
		action_prob[optimal_action] = 1		

		#If there is no optimal policy yet, we set all action probabilities to 0.25
		Qsa_max = return_stateprobabilities(i, np.argmax(Q1, axis=1)[i])
		Qsa_reward = 0
		best_move = 0
		#Loop over the data returned by return_stateprobabilities()
		for k in Qsa_max:
			Qsa_reward += (k[0] * R1[0,k[1]]) + (GAMMA * k[0] * np.max(Q1[k[1]]))
		best_move += (Qsa_reward)

		if best_move >= np.amax(Q1, axis=1)[i]:
			action_prob = [0.25, 0.25, 0.25, 0.25]	

		#Loop over actions per state
		for j in range(4):								
			state_value_update = 0
			trans_prob = return_stateprobabilities(i,j)
			term = 0
			#Loop over the data returned by return_stateprobabilities()
			for k in trans_prob:
				term += (k[0] * R1[0,k[1]]) + (GAMMA * k[0] * np.max(Q1[k[1]]))
			state_value_update += (action_prob[j] * term)
			
			Q2[i,j] = state_value_update
			
		

	#This if statement checks for converging, and prints tables.
	if np.array_equal(optimal_policy,np.argmax(Q2, axis=1)):
		print("Completed {} cycle(s)".format(amount_of_steps))
		print("The policies are identical, so the policy has converged.")
		print(Q1)
		highest_scores = np.amax(Q1, axis=1)
		optimal_policy = np.argmax(Q1, axis=1)
		optimal_policy_2 = []
		for i in optimal_policy:
			if i == 0:
				optimal_policy_2.append('up')
			elif i == 1:
				optimal_policy_2.append('right')
			elif i == 2:
				optimal_policy_2.append('down')
			elif i == 3:
				optimal_policy_2.append('left')
		for i in terminal_states:
			for j in range(16):
				if i == j:
					optimal_policy_2[j] = 'N/A'
		print("")
		print("The values of all 16 states are shown below")
		print(highest_scores[0:4])
		print(highest_scores[4:8])
		print(highest_scores[8:12])
		print(highest_scores[12:16])
		print("")
		print("The values of all 16 states are shown below")
		print(optimal_policy_2[0:4])
		print(optimal_policy_2[4:8])
		print(optimal_policy_2[8:12])
		print(optimal_policy_2[12:16])
		break

	if amount_of_steps == 100:
		print("Completed {} cycle(s)".format(amount_of_steps))
		print("Reached 100 cycles, policy iteration loop ended.")
		print(Q1)
		highest_scores = np.amax(Q1, axis=1)
		print("")
		print("The values of all 16 states are shown below")
		print(highest_scores[0:4])
		print(highest_scores[4:8])
		print(highest_scores[8:12])
		print(highest_scores[12:16])
		break
		
	#This allows us to keep track of how many loops we've done
	print("Completed {} cycle(s)".format(amount_of_steps))
	print(Q1)
	print("")
