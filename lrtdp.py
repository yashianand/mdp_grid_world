from grid_world_ssp import sspWorld, printEnvironment
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
    pi[state] = action
    v[state] = qValue(grid, state, action)

def pickNextState(grid, state, action):
    transitions = grid.get_successors(state, action)
    s_prime, prob = [], []
    for i in transitions:
        s_prime.append(i[0])
        prob.append(i[1])
    next_state = np.random.choice(s_prime, p=prob)
    return next_state

def residual(grid, state):
    action = greedy_action(grid, state)
    return abs(v[state] - qValue(grid, state, action))

def printEnvironment(grid, vals,  policy=False):
    res = ""
    for r in range(grid.rows):
        res += "|"
        for c in range(grid.cols):
            if policy:
                val = ["Left", "Up", "Right", "Down"][vals[r][c]]
            else:
                val = str([vals[r][c]])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

'''
--------------------------Check if solved--------------------------
'''
def checkSolved(grid, state, epsilon):
    rv = True
    open = []
    closed = []
    if state not in SOLVED:
        open.append(state)

    while open != []:
        state = open.pop()
        closed.append(state)

        # check residual
        if residual(grid, state) > epsilon:
            rv = False
            continue

        # expand state
        action = greedy_action(grid, state)
        for (next_state, trans_prob) in grid.get_successors(state, action):
            if trans_prob > 0:
                if (next_state not in SOLVED) and (next_state not in list(set(open) | set(closed))):
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
    return rv

'''
-------------------------------------------------------------------
'''

def lrtdp_trial(grid, state, epsilon):
    visited = []
    while state not in SOLVED:
        # insert into visited
        visited.append(state)

        # check termination at goal states
        r, c = grid.to_pos(state)
        if grid.grid_list[r][c] == grid.terminal_marker:
            break

        # pick best action and update hash
        a = greedy_action(grid, state)
        update(grid, state)

        # stochastically simulate next state
        state = pickNextState(grid, state, a)

    # try labeling visited states in reverse order
    while visited != []:
        # print("Visited: ", visited)
        state = visited.pop()
        if not checkSolved(grid, state, epsilon):
            break

def lrtdp(grid, state, epsilon):
    while state not in SOLVED:
        lrtdp_trial(grid, state, epsilon)
    print("--"*20)
    print("Iteration: ", i)
    print("Value function: \n")
    printEnvironment(grid, np.array(v[:], dtype=float).reshape(4,4), policy=False)
    print("Policy: \n")
    printEnvironment(grid, np.array(pi[:], dtype=int).reshape(4,4), policy=True)
    print("--"*20)

if __name__ == "__main__":
    nS = sspWorld.num_states
    nA = sspWorld.num_actions
    for i in range(10):
        SOLVED = []
        v = np.zeros(nS)
        pi = np.zeros(nS)
        sspWorld.reset()
        lrtdp(sspWorld, sspWorld.state, epsilon=0.1)
