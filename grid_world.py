"""
References:
(1) Intelligents Agents by Dr Fern course material
(2) OpenAI Gym's FrozenLake-v0 environment (https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)
"""

from mimetypes import init
from tkinter import Grid
import matplotlib.pyplot as plt
import numpy as np
import sys
from six import StringIO, b
from contextlib import closing

from gym.envs.toy_text import discrete
from gym import utils

# Global variables
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFTF",
        "FTTF",
        "FTFG",
        "FFFF"
        ]
    }

TransitionProb = [0.8, 0.2]

# DFS to check that it's a valid path.
def is_valid(board: list[list[str]], max_size: int) -> bool:
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size or c_new < 0 or c_new >= max_size:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "T":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """
    valid = False
    board = []
    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "T"], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
        # print(valid)
        # print(board)
    return ["".join(x) for x in board]


# Grid World Environment
class GridWorldEnv(discrete.DiscreteEnv):
    """
    Default grid world environment:
        SFTF
        FTTF
        FTFG
        FFFF

    S: Start cell
    F: Free cell, reward=-1
    T: Traffic cell, reward=-10
    G: Goal, reward=+100

    The episode ends when you reach the Goal.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4", stochastic=False):
        """
        desc: map description, list[list[char code]]
        map_name: name of the map
        stochastic: whether to use stochastic transitions
        """
        # Create the environment
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c') # to numpy array where each element is a byte (type: numpy.bytes_)
        self.nrow, self.ncol = nrow, ncol = desc.shape

        # Rewards for each state
        rew_goal = 100
        rew_step = -1
        rew_traffic = -10

        # number of actions and states
        nA = 4
        nS = nrow * ncol

        initial_state_distribution = np.array(desc == b'S').astype('float64').ravel() # 2d to 1d
        initial_state_distribution /= initial_state_distribution.sum()

        # P - transition probability matrix
        P = {s : {a : [] for a in range(nA)} for s in range(nS)} # len(P) = nS, len(P[0]) = nA


        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1, 0)
            elif a ==DOWN:
                row = min(row+1, nrow-1)
            elif a == RIGHT:
                col = min(col+1, ncol-1)
            elif a == UP:
                row = max(row-1, 0)
            return (row, col)

        def update_prob_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            done = bytes(newletter) in b'G'
            if newletter == b'G':
                rew = rew_goal
            elif newletter == b'T':
                rew = rew_traffic
            else:
                rew = rew_step
            return newstate, rew, done

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    """
                    li: list of tuples. Transition from state s to (all) states s' with action a.
                    (transition_probability, in ending in state s, with a reward, terminal state(bool))
                    """
                    letter = desc[row, col]
                    if letter in b"G":
                        li.append((1.0, s, 0, True))
                    else:
                        if stochastic:
                            for b, p in zip([a, (a+1)%4, (a+2)%4, (a+3)%4], TransitionProb):
                                newstate, rew, done = update_prob_matrix(row, col, b)
                                li.append((p, newstate, rew, done))
                        else:
                            newstate, rew, done = update_prob_matrix(row, col, a)
                            li.append((1.0, newstate, rew, done))
                            # print(s, li) # sanity check

        super().__init__(nS, nA, P, initial_state_distribution)

    def render(self, mode='human'):
        # Attributes of a GridWorldEnv instance
        # attributes = dir(self)
        # print(attributes)
        outfile = StringIO() if mode=='ansi' else sys.stdout
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


# to generate a random env, do the following:
# map_name = generate_random_map(4, 0.8)
# print(is_valid(map_name, 4)) # sanity check
# env = GridWorldEnv(desc=map_name)
# env.render()

# to use the default env, do the following:
env = GridWorldEnv()
env.render()


# For checking the environment created (game to be played by the user)
print("---------actions--------")
print("a: Left\ns: Down\nd: Right\nw: Up\n(q: quit)")
rew = 0

for _ in range(1000):
    a = input("Enter action: ")
    if a == "q":
        break
    elif a == 'a':
        a = 0
    elif a == 's':
        a = 1
    elif a == 'd':
        a = 2
    elif a == 'w':
        a = 3
    else:
        print("Invalid action")
        continue

    observation, reward, done, info = env.step(a)
    print(env.step(a))
    rew += reward
    print("---------actions--------")
    print("a: Left\ns: Down\nd: Right\nw: Up\n(q: quit)")
    print()
    print("current state:" + str(observation))
    if info['prob'] == TransitionProb[0] or info['prob'] == 1:
        print('moved to expected direction')
    else:
        print('moved to unexpected direction')
    print("probabilty: " + str(info['prob']))
    print("current reward:" + str(rew))
    print()
    env.render()
    print()
    if done:
        print('Reached Goal!')
        break
