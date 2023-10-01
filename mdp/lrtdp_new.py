from grid_world_ssp import sspWorld
import numpy as np

def greedy_action(grid, state, v):
    # Return both action and its qValue
    q_values = [qValue(grid, state, action, v) for action in range(grid.num_actions)]
    best_action = np.argmin(q_values)
    return best_action, q_values[best_action]

def qValue(grid, state, action, v):
    transitions = grid.get_successors(state, action)
    state_val = sum(trans_prob * v[next_state] for (next_state, trans_prob) in transitions)
    state_val += grid.get_reward(state)
    return state_val

def update(grid, state, v, pi):
    action, q_val = greedy_action(grid, state, v)
    pi[state] = action
    v[state] = q_val

def residual(grid, state, v):
    _, q_val = greedy_action(grid, state, v)
    return abs(v[state] - q_val)

def pickNextState(grid, state, action):
    transitions = grid.get_successors(state, int(action))
    s_prime = [i[0] for i in transitions]
    prob = [i[1] for i in transitions]
    return np.random.choice(s_prime, p=prob)

def checkSolved(grid, state, epsilon, SOLVED, v, pi):
    open_set = {state}
    closed_set = set()

    while open_set:
        state = open_set.pop()
        closed_set.add(state)

        if residual(grid, state, v) <= epsilon:
            action, _ = greedy_action(grid, state, v)
            successors = {next_state for (next_state, trans_prob) in grid.get_successors(state, action) if trans_prob > 0}
            open_set.update(successors - SOLVED - closed_set - open_set)
        else:
            while closed_set:
                state = closed_set.pop()
                update(grid, state, v, pi)
            return False

    SOLVED.update(closed_set)
    return True

def lrtdp_trial(grid, state, epsilon, SOLVED, v, pi):
    visited = []
    while state not in SOLVED:
        visited.append(state)
        r, c = grid.to_pos(state)
        if grid.grid_list[r][c] == grid.terminal_marker:
            break
        update(grid, state, v, pi)
        action = pi[state]  # use policy directly
        state = pickNextState(grid, state, action)
    while visited:
        state = visited.pop()
        if not checkSolved(grid, state, epsilon, SOLVED, v, pi):
            break

def printEnvironment(grid, vals,  policy=False):
    res = ""
    for r in range(grid.rows):
        res += "|"
        for c in range(grid.cols):
            if policy:
                val = ["Left", "Up", "Right", "Down"][int(vals[r][c])]
            else:
                val = str([vals[r][c]])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

def lrtdp(grid, state, epsilon=0.1, max_trials=100):
    trial = 1
    nS = grid.num_states
    v = np.zeros(nS)
    pi = np.zeros(nS)
    SOLVED = set()
    grid.reset()
    while state not in SOLVED and trial <= max_trials:
        print('trial: ', trial)
        lrtdp_trial(grid, state, epsilon, SOLVED, v, pi)
        trial += 1
        printEnvironment(grid, pi.reshape(15, 15), policy=True)

    # Uncomment for debugging or visualization
    # print("--" * 20)
    # print("Value function:\n")
    # printEnvironment(grid, v.reshape(4, 4), policy=False)
    # print("Policy:\n")
    # printEnvironment(grid, pi.reshape(4, 4), policy=True)
    # print("--" * 20)

# Commenting out the execution for now
lrtdp(sspWorld, sspWorld.state, epsilon=0.1)

# Return the refactored functions for review
# greedy_action, qValue, update, residual, checkSolved, lrtdp_trial, lrtdp
