from flat2factoredRep import gridWorld, printEnvironment
import numpy as np

def valueIteration(grid, gamma=0.99, epsilon=0.01):
    nA = grid.num_actions
    nRows = grid.rows
    nCols = grid.cols

    v_new = np.zeros((nRows, nCols))
    pi_new = np.zeros((nRows, nCols))

    factoredStates = grid.getStateFactorRep()
    print("Factored States: ")
    for i in factoredStates:
        print(i)

    while True:
        v = v_new.copy()
        delta = 0
        for i in range(len(factoredStates)):
            state, traffic, goal = factoredStates[i]
            state_action_pair = []
            value = []
            if goal == True:
                v_new[state[0]][state[1]] = grid.get_reward(factoredStates[i])
                continue
            for action in range(nA):
                factoredAction = grid.getActionFactorRep(action)
                state_action_pair.append((state, factoredAction))
                value.append(sum([trans_prob*(v[next_state[0][0]][next_state[0][1]]) for (next_state, trans_prob) in grid.get_successors(state, action)]) )
            v_new[state[0]][state[1]] = grid.get_reward(factoredStates[i]) + gamma * max(value)
            pi_new[state[0]][state[1]] = value.index(max(value))
            delta = max(delta, abs((v_new[state[0]][state[1]]) - (v[state[0]][state[1]])))

        if delta < epsilon:
            return v_new, pi_new



v, pi = valueIteration(gridWorld, gamma=0.99)
print("\nFinal State Values: \n")
printEnvironment(np.array(v[:]).reshape((4,4)))
print("\nFinal Policy:\n")
printEnvironment(np.array(pi[:], dtype=int).reshape((4,4)), policy=True)
