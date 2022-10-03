import numpy as np
from pip import main

class FactoredGridWorld:
    gridWorld = None
    def __init__(self, grid, directions, state=(0, 0), terminal_marker='G'):
        self.reset()
        self.grid = grid = np.asarray(grid, dtype='c')
        self.grid_list = [[c.decode('utf-8') for c in line] for line in self.grid.tolist()]
        self.num_actions = len(directions)
        self.state = state
        self.terminal_marker = terminal_marker
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_states = self.rows * self.cols

    def reset(self):
        self.state = 0

    # should I code for all possible states and weed out unreachable states later?
    def getStateFactorRep(self):
        featureRep = [] # records <int x, int y, bool traffic, bool goal>
        for i in range(self.rows):
            for j in range(self.cols):
                currState = self.grid_list[i][j]
                if currState == self.terminal_marker:
                    featureRep.append([(i, j), False, True])
                elif currState == 'T':
                    featureRep.append([(i, j), True, False])
                else:
                    featureRep.append([(i, j), False, False])

        return featureRep

    def step(self, action):
        terminal = False
        factoredNextStates = self.get_successors(self.state, action)
        s_prime, prob = [], []
        for i in factoredNextStates:
            s_prime.append(i[0])
            prob.append(i[1])
        next_state = np.random.choice(s_prime, p=prob)
        self.state = next_state[0]
        idx = s_prime.index(next_state)
        reward = self.get_reward(next_state)
        r, c = next_state[0][0], next_state[0][1]
        if self.grid_list[r][c] == self.terminal_marker:
            terminal = True
        return next_state, reward, prob[idx], terminal

    def getActionFactorRep(self, a):
        # a = 0: left, a = 1: up, a = 2: right, a = 3: down
        if a == 0:
            return actions[0]
        elif a == 1:
            return actions[1]
        elif a == 2:
            return actions[2]
        else:
            return actions[3]

    def is_boundary(self, state):
        x, y = state
        return (x < 0 or x > self.rows-1 or y < 0 or y > self.cols-1 )

    def move(self, state, action):
        new_state = tuple(x + y for (x, y) in zip(state, self.getActionFactorRep(action)))
        if self.is_boundary(new_state):
            self.state = state
            return self.state, True
        else:
            self.state = new_state
            return self.state, False


    def getTransitionFactorRep(self, s1, a, s2):
        # s = (x, y), a = (x, y)
        new_state_pos, is_wall = self.move(s1, a)

        success_prob = 0.8
        fail_prob = 0.2

        # if desired action leads into the boundary
        if is_wall:
            # print("hit boundary")
            if(s1 == s2):
                success_prob = 1
                return success_prob

        # if desired action is viable
        elif not is_wall:
            if(s2 == new_state_pos):
                return success_prob
            # if desired action fails (with a prob=0.2)
            if(s2 == s1):
                return fail_prob

        return 0

    def get_successors(self, state, action):
        successors = []
        for i in range(self.num_states):
            next_state = self.getStateFactorRep()[i][0]
            p = self.getTransitionFactorRep(state, action, next_state)
            if p > 0:
                successors.append((self.getStateFactorRep()[i], p))
        return successors

    def get_reward(self, factoredStateRep):
        (x, y), traffic, goal = factoredStateRep
        if goal == True:
            pos_reward = 100
        elif traffic == True:
            pos_reward = -10
        else:
            pos_reward = -1
        return pos_reward

# Visualization
def printEnvironment(grid, policy=False):
    res = ""
    for r in range(gridWorld.rows):
        res += "|"
        for c in range(gridWorld.cols):
            if policy:
                val = ["Left", "Up", "Right", "Down"][grid[r][c]]
            else:
                val = str(grid[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)









# grid_rewards = {
#     'F': -1,
#     'T': -10,
#     'G': 100
# }


def readMDPfile(filename):
    """
    Input : MDP filename
    Output :
        grid_list : [list of chars] MDP grid world
        nrows : [int] number of rows
        ncols : [int] number of columns
    """
    mdpFile = open(filename, 'r')
    readMDP = mdpFile.read()
    grid = readMDP.replace('\n', '').split(',')
    mdpFile.close()
    return grid


filename = 'gridMap.txt'

actions = {
    0: (0, -1), # left
    1: (-1, 0), # up
    2: (0, 1), # right
    3: (1, 0) # down
}

directions = [0, 1, 2, 3]

grid = readMDPfile(filename)
gridWorld = FactoredGridWorld(grid, directions, terminal_marker='G')
