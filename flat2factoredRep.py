import numpy as np
from pip import main

class FactoredGridWorld:
    gridWorld = None
    def __init__(self, grid, directions, terminal_state, state=[(0, 0), False]):
        self.reset()
        self.grid = grid = np.asarray(grid, dtype='c')
        self.grid_list = [[c.decode('utf-8') for c in line] for line in self.grid.tolist()]
        self.num_actions = len(directions)
        self.state = state
        self.terminal_state = terminal_state
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_states = self.rows * self.cols

    def reset(self):
        self.state = [(0, 0), False]

    def getStateFactorRep(self):
        featureRep = [] # records [(int x, int y), bool traffic]
        for i in range(self.rows):
            for j in range(self.cols):
                currState = self.grid_list[i][j]
                if currState == 'T':
                    featureRep.append([(i, j), True])
                else:
                    featureRep.append([(i, j), False])

        return featureRep

    def step(self, action):
        terminal = False
        factoredNextStates = self.get_successors(self.state, action)
        s_prime, prob = [], []
        print("factoredNextStates: ", factoredNextStates)
        for i in factoredNextStates:
            s_prime.append(i[0])
            prob.append(i[1])
        print("s_prime: {}, prob: {}".format(s_prime, prob))
        next_state_idx = np.random.choice(len(s_prime), p=prob)
        print("next_state_idx: ", next_state_idx)
        self.state = s_prime[next_state_idx]
        # idx = s_prime.index(next_state)
        reward = self.get_reward(self.state)
        if self.is_goal(self.state):
            terminal = True
        print("s_prime[next_state_idx] : {}, reward: {}, prob[next_state_idx]: {}, terminal: {}".format(s_prime[next_state_idx], reward, prob[next_state_idx], terminal))
        return s_prime[next_state_idx], reward, prob[next_state_idx], terminal

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

    def is_goal(self, state):
        return state == self.terminal_state

    def move(self, currFactoredState, action):
        state, traffic = currFactoredState
        new_state = tuple(x + y for (x, y) in zip(state, self.getActionFactorRep(action)))
        if self.is_boundary(new_state):
            self.state = currFactoredState
            return self.state, True
        else:
            if self.grid_list[new_state[0]][new_state[1]] == 'T':
                self.state = [new_state, True]
            else:
                self.state = [new_state, False]
            return self.state, False


    # def getTransitionFactorRep(self, s1, a, s2):
    #     '''
    #     s1: current state, format: [(int x, int y), bool traffic]
    #     a: action, format: [int a]
    #     s2: next state, format: [(int x, int y), bool traffic]
    #     '''
    #     new_factored_state, is_wall = self.move(s1, a)
    #     print("new_factored_state: ", new_factored_state)

    #     success_prob = 0.8
    #     fail_prob = 0.2

    #     # if desired action leads into the boundary
    #     if is_wall:
    #         # print("hit boundary")
    #         if(s1 == s2):
    #             success_prob = 1
    #             traffic = s1[1]
    #             traffic_prob = 1 if traffic else 0
    #             return np.prod([success_prob, traffic_prob])

    #     # if desired action is viable
    #     elif not is_wall:
    #         print("here*********************************************")
    #         print("s2: {}, new_factored_state: {}".format(s2, new_factored_state))
    #         if(s2 == new_factored_state):
    #             traffic = new_factored_state[1]
    #             traffic_prob = 1 if traffic else 0
    #             print("Prob: ", np.prod([success_prob, traffic_prob]))
    #             return np.prod([success_prob, traffic_prob])
    #         # if desired action fails (with a prob=0.2)
    #         if(s2 == s1):
    #             traffic = s1[1]
    #             traffic_prob = 1 if traffic else 0
    #             return np.prod([fail_prob, traffic_prob])

    #     return 0

    def getTransitionFactorRep(self, curr_state, action, next_state):
        '''
        curr_state: current state, format: [(int x, int y), bool traffic]
        action: [int a]
        next_state: [(int x, int y), bool traffic]
        '''
        succ_factored_state, is_wall = self.move(curr_state, action)
        # print("\ncurr_state: {}, next_state: {}, succ_factored_state: {}".format(curr_state, next_state, succ_factored_state))

        success_prob = 0.8
        fail_prob = 0.2

        if is_wall:
            # print("hit boundary")
            transition_probs = []
            for feature_idx in range(len(curr_state)):
                if (curr_state[feature_idx] == next_state[feature_idx]):
                    transition_probs.append(1)
                else:
                    transition_probs.append(0)
            return np.prod(transition_probs)

        elif not is_wall:
            # print("no boundary")
            transition_probs = []
            if (next_state[0]==succ_factored_state[0]):
                transition_probs.append(success_prob)
                if (next_state[1]==succ_factored_state[1]):
                    transition_probs.append(1)
                elif (next_state[1]!=succ_factored_state[1]):
                    transition_probs.append(0)

            elif (next_state[0]==curr_state[0]):
                transition_probs.append(fail_prob)
                if (next_state[1]==curr_state[1]):
                    transition_probs.append(1)
                elif (next_state[1]!=curr_state[1]):
                    transition_probs.append(0)

            else:
                return 0

            return np.prod(transition_probs)

    def get_successors(self, state, action):
        successors = []
        factored_states = self.getStateFactorRep()
        for i in range(self.num_states):
            next_state = factored_states[i]
            p = self.getTransitionFactorRep(state, action, next_state)
            if p > 0:
                successors.append((next_state, p))
        return successors

    def get_reward(self, factoredStateRep):
        (x,y), traffic = factoredStateRep
        if self.is_goal((x,y)) == True:
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
gridWorld = FactoredGridWorld(grid, directions, terminal_state=[(2, 3), False])
gridWorld.reset()
gridWorld.step(action=2)
