import numpy as np
class GridWorld:
    def __init__(self, grid, rewards, directions, state=0, terminal_marker='G'):
        self.grid = np.asarray(grid, dtype='c')
        self.grid_list = [[c.decode('utf-8') for c in line] for line in self.grid.tolist()]
        self.rewards = rewards
        self.terminal_marker = terminal_marker
        self.actions = [actions[d] for d in directions]
        self.num_actions = len(self.actions)
        self.state = state
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_states = self.rows * self.cols
        self.reset()

    def reset(self):
        self.state = 0

    def to_s(self, row, col):
        return row * self.cols + col

    def to_pos(self, state):
        return state // self.rows, state % self.cols

    def is_boundary(self, pos):
        x, y = pos
        return (x < 0 or x >= self.rows or y < 0 or y >= self.cols)

    def move(self, state, action):
        state_pos = np.array(self.to_pos(state))
        new_state_pos = tuple((state_pos + self.actions[action]).reshape(1, -1)[0])
        if self.is_boundary(new_state_pos):
            return state, True
        else:
            return self.to_s(new_state_pos[0], new_state_pos[1]), False

    def get_possible_next_states(self, state):
        possible_states = set()
        for action in range(self.num_actions):
            next_state, _ = self.move(state, action)
            possible_states.add(next_state)
        return possible_states

    def get_side_states(self, state, action):
        side_states = []
        for a in range(self.num_actions):
            if (a!=action):
                new_state, is_wall = self.move(state, a)
                side_states.append(new_state)
        return side_states

    def get_transition_prob(self, s1, action, s2):
        new_state, is_wall = self.move(s1, action)
        sstates = self.get_side_states(s1, action)

        success_prob = 0.7
        fail_prob = 0.1

        if is_wall:
            if(s1 == s2):
                success_prob = 1
                return success_prob

        elif not is_wall:
            if(s2 == new_state):
                return success_prob

            for side_state in sstates:
                if(s2 == side_state):
                    state_count = sstates.count(s2)
                    fail_prob *= state_count
                    return fail_prob

        return 0

    def get_successors(self, state, action):
        successors = []
        for next_state in self.get_possible_next_states(state):
            p = self.get_transition_prob(state, action, next_state)
            if p > 0:
                successors.append((next_state, p))
        return successors

    def get_reward(self, state):
        state_pos = self.to_pos(state)
        x, y = state_pos[0], state_pos[1]
        return self.rewards[self.grid_list[x][y]]


actions = {
    0: np.array([0, -1]), # left
    1: np.array([-1, 0]), # up
    2: np.array([0, 1]), # right
    3: np.array([1, 0]) # down
}
# grid =[
#         "SFTF",
#         "FTTF",
#         "FTFG",
#         "FFFF"
#         ]

grid = [
    "SFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFF",
    "FFTTTFFFFFTTTFF",
    "FFTTTFFFFFTTTFF",
    "FFTTTFFFFFTTTFF",
    "FFFFFTTTTTFFFFF",
    "FFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFF",
    "FFFFFFTTTTFFFFF",
    "FFFFFTTTTTTFFFF",
    "FFFFFFFFFFFFFFF",
    "FFFFFFFFFFFFFFF",
    "FFFFFFFTTFFFFFF",
    "FFFFFFFTTFFFFFF",
    "FFFFFFFFFFFFFFG"
]

grid_rewards = {
    'F': -1,
    'T': -10,
    'G': 100
}

ssp_cost = {
    'S': 1,
    'F': 1,
    'T': 10,
    'G': 0
}

directions = [0, 1, 2, 3] # [left, up, right, down]

sspWorld = GridWorld(grid, ssp_cost, directions, terminal_marker='G')
