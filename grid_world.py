import matplotlib.pyplot as plt
import numpy as np

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


print(is_valid(MAPS["4x4"], 4))
