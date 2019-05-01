import numpy as np


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