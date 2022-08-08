from grid_world import gridWorld, printEnvironment
import numpy as np

def valueIteration(grid, gamma=0.99, epsilon=0.01):
    nS = grid.num_states
    nA = grid.num_actions

    v_new = np.zeros(nS)
    pi_new = np.zeros(nS)

    while True:
        v = v_new.copy()
        pi = pi_new.copy()
        delta = 0
        for state in range(nS):
            state_action_pair = []
            r, c = grid.to_pos(state)
            value = []
            if grid.grid_list[r][c] == grid.terminal_marker:
                v_new[state] = grid.rewards['G']
                continue
            for action in range(nA):
                state_action_pair.append((state, action))
                value.append(sum([trans_prob*v[next_state] for (next_state, trans_prob) in grid.get_successors(state, action)]) )
            v_new[state] = grid.get_reward(state) + gamma * max(value)
            pi_new[state] = value.index(max(value))
            delta = max(delta, abs(v_new[state] - v[state]))

        # print("State Value: \n{} \nPolicy: \n{}".format(np.array(v_new[:]).reshape((4,4)), np.array(pi[:]).reshape((4,4))))
        # input("press Enter")

        if delta < epsilon:
            return v_new, pi_new



# v, pi = valueIteration(gridWorld, gamma=0.99)
# print("\nFinal State Values: \n")
# printEnvironment(np.array(v[:]).reshape((4,4)))
# print("\nFinal Policy:\n")
# printEnvironment(np.array(pi[:], dtype=int).reshape((4,4)), policy=True)
