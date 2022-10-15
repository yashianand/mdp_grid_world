from fmdp.flat2factoredRep import gridWorld, printEnvironment
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
        pi = pi_new.copy()
        delta = 0
        for state in factoredStates:
            (x,y), traffic = state
            state_action_pair = []
            value = []
            if grid.is_goal(state) == True:
                v_new[x][y] = grid.get_reward(state)
                continue
            for action in range(nA):
                factoredAction = grid.getActionFactorRep(action)
                state_action_pair.append((state, factoredAction))
                value.append(sum([trans_prob*(v[next_state[0][0]][next_state[0][1]]) for (next_state, trans_prob) in grid.get_successors(state, action)]) )
            v_new[x][y] = grid.get_reward(state) + gamma * max(value)
            pi_new[x][y] = value.index(max(value))
            delta = max(delta, abs((v_new[x][y]) - (v[x][y])))

        # print("State Value: \n{} \nPolicy: \n{}".format(np.array(v_new[:]).reshape((4,4)), np.array(pi[:]).reshape((4,4))))
        # input("press Enter")

        if delta < epsilon:
            return v_new, pi_new



v, pi = valueIteration(gridWorld, gamma=0.99)
print("\nFinal State Values: \n")
printEnvironment(np.array(v[:]).reshape((4,4)))
print("\nFinal Policy:\n")
printEnvironment(np.array(pi[:], dtype=int).reshape((4,4)), policy=True)
