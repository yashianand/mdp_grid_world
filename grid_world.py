"""
References:
(1) Intelligents Agents by Dr Fern course material
(2) OpenAI Gym's FrozenLake-v0 environment (https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)
"""

from tkinter import Grid
import matplotlib.pyplot as plt
import numpy as np
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

map_name = generate_random_map(4, 0.8)

# print(is_valid(MAPS["4x4"], 4))


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

    def __init__(self, desc=None, map_name="4x4", stochastic=True):
        """
        desc: map description, list[list[char code]]
        map_name: name of the map
        stochastic: whether to use stochastic transitions
        """

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        # number of actions and states
        nA = 4
        nS = nrow * ncol

        # P - transition probability matrix
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        self.TransitProb = np.zeros((nA, nS + 1, nS + 1))
        self.TransitReward = np.zeros((nS + 1, nA))

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

        
GridWorldEnv(map_name)
