import numpy as np
class GridWorld:
    gridWorld = None

    def __init__(self, grid, rewards, directions, state=0, terminal_marker='G'):
        self.reset()
        self.grid = grid = np.asarray(grid, dtype='c')
        # self.grid_list = self.grid.tolist()
        self.grid_list = [[c.decode('utf-8') for c in line] for line in self.grid.tolist()]
        self.rewards = rewards
        self.terminal_marker = terminal_marker
        self.actions = [actions[d] for d in directions]
        self.num_actions = len(self.actions)
        self.state = state
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.num_states = self.rows * self.cols

    def reset(self):
        self.state = 0

    def step(self, action):
        terminal = False
        # reward = self.get_reward(self.state)
        transitions = self.get_successors(self.state, action)
        s_prime, prob = [], []
        for i in transitions:
            s_prime.append(i[0])
            prob.append(i[1])
        next_state = np.random.choice(s_prime, p=prob)
        self.state = next_state
        idx = s_prime.index(next_state)
        reward = self.get_reward(self.state)
        r, c = self.to_pos(next_state)
        if self.grid_list[r][c] == self.terminal_marker:
            terminal = True
        return next_state, reward, prob[idx], terminal

    def to_s(self, row, col):
        self.state = row * self.cols + col
        return self.state

    def to_pos(self, state):
        row, col = state // self.rows, state % self.cols
        return row, col

    def is_boundary(self, pos):
        x, y = pos
        return (x < 0 or x > self.rows-1 or y < 0 or y > self.cols-1 )

    def move(self, state, action):
        state_pos = self.to_pos(state)
        new_state_pos = tuple((state_pos + self.actions[action]).reshape(1, -1)[0])
        if self.is_boundary(new_state_pos):
            self.state = state
            return self.state, True
        else:
            self.state = self.to_s(new_state_pos[0], new_state_pos[1])
            return self.to_pos(self.state), False

    def get_transition_prob(self, s1, action, s2):
        '''
        TODO:
            - convert position (single int) to state (x,y coords)
        '''
        new_state_pos, is_wall = self.move(s1, action)
        s1_pos, s2_pos = self.to_pos(s1), self.to_pos(s2)
        # print("s1: {}, a: {}, s2: {}, new_state: {}".format(s1_pos, action, s2_pos, new_state_pos))

        success_prob = 0.8
        fail_prob = 0.2

        # if desired action leads into the boundary
        if is_wall:
            # print("hit boundary")
            if(s1_pos == s2_pos):
                success_prob = 1
                return success_prob

        # if desired action is viable
        elif not is_wall:
            if(s2_pos == new_state_pos):
                return success_prob
            # if desired action fails (with a prob=0.2)
            if(s2_pos == s1_pos):
                return fail_prob

        return 0

    def get_successors(self, state, action):
        successors = []
        for next_state in range(self.num_states):
            p = self.get_transition_prob(state, action, next_state)
            if p > 0:
                successors.append((next_state, p))
        return successors

    def get_reward(self, state):
        # print("\nstate: {}, action: {}".format(state, action))
        # reward = {}
        # successors = self.get_successors(state, action)
        # for s in successors:
            # new_state = s[0]
        # print("get_reward() new_state: ", new_state)
        state_pos = self.to_pos(state)
        x, y = state_pos[0], state_pos[1]
        if self.grid_list[x][y] == 'S':
            pos_reward = 1
        else:
            pos_reward = self.rewards[self.grid_list[x][y]]
            # reward[new_state] = pos_reward
        return pos_reward

# Visualization
def printEnvironment(grid, policy=False):
    res = ""
    for r in range(grid.rows):
        res += "|"
        for c in range(grid.cols):
            if policy:
                val = ["Left", "Up", "Right", "Down"][grid[r][c]]
            else:
                val = str(grid[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)


actions = {
    0: np.array([0, -1]), # left
    1: np.array([-1, 0]), # up
    2: np.array([0, 1]), # right
    3: np.array([1, 0]) # down
}
grid =[
        "SFTF",
        "FTTF",
        "FTFG",
        "FFFF"
        ]

grid_rewards = {
    'F': -1,
    'T': -10,
    'G': 100
}

ssp_cost = {
    'F': 1,
    'T': 10,
    'G': 0
}

directions = [0, 1, 2, 3] # [left, up, right, down]

# gridWorld = GridWorld(grid, grid_rewards, directions, terminal_marker='G')
sspWorld = GridWorld(grid, ssp_cost, directions, terminal_marker='G')
