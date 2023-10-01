from grid_world_ssp import sspWorld, printEnvironment
import numpy as np

'''
----------------------------Helper Functions------------------------
'''
def greedy_action(grid, state, v):
    q_values = []
    for action in range(grid.num_actions):
        q_values.append(qValue(grid, state, action, v))
    return np.argmin(q_values)

def qValue(grid, state, action, v):
    state_val = sum([trans_prob*v[next_state] for (next_state, trans_prob) in grid.get_successors(state, action)])
    state_val += grid.get_reward(state)
    return state_val

def update(grid, state, v, pi):
    action = greedy_action(grid, state, v)
    pi[state] = action
    v[state] = qValue(grid, state, action, v)

def pickNextState(grid, state, action):
    transitions = grid.get_successors(state, action)
    s_prime, prob = [], []
    for i in transitions:
        s_prime.append(i[0])
        prob.append(i[1])
    next_state = np.random.choice(s_prime, p=prob)
    return next_state

def residual(grid, state, v):
    action = greedy_action(grid, state, v)
    return abs(v[state] - qValue(grid, state, action, v))

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
def checkSolved(grid, state, epsilon, SOLVED, v, pi):
    rv = True
    open = []
    closed = []
    if state not in SOLVED:
        open.append(state)

    while open != []:
        state = open.pop()
        closed.append(state)

        # check residual
        if residual(grid, state, v) > epsilon:
            rv = False
            continue

        # expand state
        action = greedy_action(grid, state, v)
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
            update(grid, state, v, pi)
    return rv

'''
-------------------------------------------------------------------
'''

def lrtdp_trial(grid, state, epsilon, SOLVED, v, pi):
    print('here')
    visited = []
    while state not in SOLVED:
        # insert into visited
        visited.append(state)

        # check termination at goal states
        r, c = grid.to_pos(state)
        if grid.grid_list[r][c] == grid.terminal_marker:
            break

        # pick best action and update hash
        a = greedy_action(grid, state, v)
        update(grid, state, v, pi)

        # stochastically simulate next state
        state = pickNextState(grid, state, a)
    print('here 2')
    # try labeling visited states in reverse order
    while visited != []:
        # print("Visited: ", visited)
        state = visited.pop()
        if not checkSolved(grid, state, epsilon, SOLVED, v, pi):
            break

def lrtdp(grid, state, epsilon=0.1):
    nS = grid.num_states
    v = np.zeros(nS)
    pi = np.zeros(nS)
    for i in range(100):
        print('iteration: ', i)
        SOLVED = []

        grid.reset()
        while state not in SOLVED:
            lrtdp_trial(grid, state, epsilon, SOLVED, v, pi)
        print("--"*20)
        print("Iteration: ", i)
        print("Value function: \n")
        printEnvironment(grid, np.array(v[:], dtype=float).reshape(4,4), policy=False)
        print("Policy: \n")
        printEnvironment(grid, np.array(pi[:], dtype=int).reshape(4,4), policy=True)
        print("--"*20)

lrtdp(sspWorld, sspWorld.state, epsilon=0.1)
