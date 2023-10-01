import numpy as np

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

# Generate random grid world
def generate_random_grid(size=8, p=0.32):
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
        board = np.random.choice(["T", "."], (size, size), p=[p, 1 - p])
        board[0][0] = "S"
        # random row and column for goal
        # row = np.random.randint(0, size)
        # col = np.random.randint(0, size)
        # board[row][col] = "G"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
        # print(valid)
        # print(board)
    return ["".join(x) for x in board]

env = generate_random_grid(size=4, p=0.45)
print(env)
# ['STT....T.T', '.T.T.TT...', '...T...GT.', 'TT.....TTT', 'T.TTTTTT..', 'T.T.T.TT..', '..TTT..T..', 'T.TTT.TT.T', 'T.T.....TT', '.T..T..TT.'] - 40% random goal
# ['S..T.T.TT.', 'TT...TT...', '..T...TT..', 'T..T.TT.T.', '..T..T.TT.', 'T..T....TT', '.TT..T...T', 'TTT..G..T.', '.T..T...T.', '.T....T...'] - 35% random goal
# ['S......T..', '..T....T..', '...TT..T.T', 'TTT.......', '..........', 'T..T..T...', '..T....T.T', '.......TTT', '..TTT.T...', '.T..G.T...'] - 30% random goal
# ['STTTT.TTTT', '..T.TTTTT.', '..T.T...T.', '.TT..T.T.T', '..........', '..TTTT.TTT', 'TT.T...TT.', 'T....TTT.T', '..T.T.T.T.', '..T......G'] - 40%
