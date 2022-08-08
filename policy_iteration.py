from grid_world import gridWorld, printEnvironment
import numpy as np

def policyIteration(grid, gamma=0.99, epsilon=0.01, max_iter=1000):
    nS = grid.num_states
    nA = grid.num_actions

    v_new = np.zeros(nS)
    pi_new = np.zeros(nS)

    for _ in range(max_iter):
        policy_stable = True

        # Policy Evaluation
        for _ in range(max_iter):
            v = v_new.copy()
            delta = 0

            for state in range(nS):
                value = 0
                action = int(pi_new[state])

                r, c = grid.to_pos(state)
                if grid.grid_list[r][c] == grid.terminal_marker:
                    v_new[state] = grid.rewards['G']
                    continue

                value += sum([trans_prob*v[next_state] for (next_state, trans_prob) in grid.get_successors(state, action)])
                v_new[state] = grid.get_reward(state) + gamma * value

                delta = max(delta, abs(v_new[state] - v[state]))

            if delta < epsilon:
                break

        # Policy Iteration
        for state in range(nS):
            max_val = grid.get_reward(state)
            for action in range(nA):
                value = 0
                for next_state, trans_prob in grid.get_successors(state, action):
                    value += trans_prob * (gamma * v_new[next_state])

                if value > max_val and pi_new[state] != action:
                    pi_new[state] = action
                    max_val = value
                    policy_stable = False

        if policy_stable:
            break
    return v_new, pi_new

# v, pi = policyIteration(gridWorld, gamma=0.99)
# print("\nFinal State Values: \n")
# printEnvironment(np.array(v[:]).reshape((4,4)))
# print("\nFinal Policy:\n")
# printEnvironment(np.array(pi[:], dtype=int).reshape((4,4)), policy=True)
