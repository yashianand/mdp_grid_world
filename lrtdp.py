from grid_world import sspWorld
import numpy as np

'''
----------------------------Helper Functions------------------------
'''
def greedy_action(grid, state):
    q_values = []
    for action in range(nA):
        q_values.append(qValue(grid, state, action))
    return np.argmin(q_values)

def qValue(grid, state, action):
    state_val = sum([trans_prob*v[next_state] for (next_state, trans_prob) in grid.get_successors(state, action)])
    state_val += grid.get_reward(state)
    return state_val

def update(grid, state):
    action = greedy_action(grid, state)
    v[state] = qValue(grid, state, action)

def pickNextState(grid, state, action):
    successors = grid.get_successors(state, action)
    for i in successors:
        next_state, trans_prob = i
    return next_state

def residual(grid, state):
    action = greedy_action(grid, state)
    return abs(v[state] - qValue(grid, state, action))

'''
--------------------------Check if solved--------------------------
'''
def checkSolved(grid, state, epsilon):
    print("*******************check solved*******************")
    rv = True
    open = []
    closed = []
    if state not in SOLVED:
        open.append(state)

    while open != []:
        state = open.pop()
        closed.append(state)

        # check residual
        if residual(grid, state) < epsilon:
            rv = False
            continue

        # expand state
        action = greedy_action(grid, state)
        for (next_state, trans_prob) in grid.get_successors(state, action):
            if trans_prob > 0:
                if (next_state not in SOLVED) and (next_state not in open.union(closed)):
                    open.append(next_state)

    if rv == True:
        # label relevant states
        for state_prime in closed:
            SOLVED.append(state_prime)
    else:
        # update states with residuals and ancestors
        while closed != []:
            state = closed.pop()
            update(grid, state)
    print("SOLVED: ", SOLVED)
    return rv

'''
-------------------------------------------------------------------
'''

def lrtdp_trial(grid, state, epsilon):
    visited = []
    while state not in SOLVED:
        # insert into visited
        print("state: ", state)
        visited.append(state)

        # check termination at goal states
        r, c = grid.to_pos(state)
        if grid.grid_list[r][c] == grid.terminal_marker:
            break

        # pick best action and update hash
        a = greedy_action(grid, state)
        update(grid, state)
        print("Best action: ", a)
        print("Grid update: ", v)

        # stochastically simulate next state
        state = pickNextState(grid, state, a)
        input()


    # try labeling visited states in reverse order
    while visited != []:
        state = visited.pop()
        if not checkSolved(grid, state, epsilon):
            break

def lrtdp(grid, state, epsilon):
    while state not in SOLVED:
        lrtdp_trial(grid, state, epsilon)
    print("Final value: ", v)

if __name__ == "__main__":
    SOLVED = []
    nS = sspWorld.num_states
    nA = sspWorld.num_actions
    v = np.zeros(nS)
    sspWorld.reset()
    lrtdp(sspWorld, sspWorld.state, epsilon=0.99)
