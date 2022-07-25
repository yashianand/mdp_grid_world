"""
References:
(1) Intelligents Agents by Dr Fern course material
(2) OpenAI Gym's FrozenLake-v0 environment (https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py)
"""

from copy import deepcopy
from mimetypes import init
from tkinter import Grid
from matplotlib import pyplot as plt
import numpy as np
import sys
from six import StringIO, b
from contextlib import closing
import time

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

TransProb = [0.8, 0.2]

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
        self.reward_range = (0, 1)

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

        # transition prob. and reward for each state-action pair (for VI)
        self.TransitionProb = np.zeros((nA, nS+1, nS+1)) # (action, start_state, end_state)
        self.TransitionReward = np.zeros((nS+1, nA))

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

                    li is used by step() to select the next state by random sampling
                    """
                    letter = desc[row, col]
                    if letter in b"G":
                        li.append((1.0, s, 100, True))
                        self.TransitionProb[a, s, nS] = 1.0
                        self.TransitionReward[s, a] = rew_goal
                    else:
                        if stochastic:
                            for b, p in zip([a, (a+1)%4, (a+2)%4, (a+3)%4], TransProb):
                                newstate, rew, done = update_prob_matrix(row, col, b)
                                li.append((p, newstate, rew, done))
                                self.TransitionProb[a, s, newstate] += p
                                self.TransitionReward[s, a] = rew_step
                        else:
                            newstate, rew, done = update_prob_matrix(row, col, a)
                            li.append((1.0, newstate, rew, done))
                            self.TransitionProb[a, s, newstate] += 1.0
                            self.TransitionReward[s, a] = rew_step
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
    def GetSuccessors(self, state, action):
        """
        Returns the possible next states (with
        non-zero transition probability) and
        their transition probabilities. this allows us to ignore all states with
        zero transition probability
        """
        next_states = np.nonzero(self.TransitionProb[action, state, :])
        probs = self.TransitionProb[action, state, next_states]
        return [(state, prob) for state, prob in zip(next_states[0], probs[0])]

    def GetReward(self, state, action):
        return self.TransitionReward[state, action]

    def GetStateSpace(self):
        return self.TransitionProb.shape[1] # = nS

    def GetActionSpace(self):
        return self.TransitionProb.shape[0] # = nA

# to generate a random env, do the following:
# map_name = generate_random_map(4, 0.8)
# print(is_valid(map_name, 4)) # sanity check
# env = GridWorldEnv(desc=map_name)
# env.render()

# to use the default env, do the following:
map_size = 4
env = GridWorldEnv()
env.render()

"""
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
"""

def value_iteration(env, beta=0.99, epsilon=0.0001):
    """
    Value Iteration Algorithm.
    Inputs:
        env: OpenAI environment.
        beta: discount factor.
        epsilon: tolerance.
    Outputs:
        V: value function.
        policy: policy function.
    """
    nS = env.GetStateSpace()
    nA = env.GetActionSpace()

    v = np.zeros(nS)
    pi = np.zeros(nS)

    v_new = np.zeros(nS)
    pi_new = np.zeros(nS)

    bellman_err = np.inf

    while bellman_err > epsilon:
        bellman_err = 0
        for state in range(nS):
            rewards = []
            state_action_pair = []
            for action in range(nA):
                total_trans_prob = 0
                state_action_pair.append((state, action))
                for next_state, trans_prob in env.GetSuccessors(state, action):
                    total_trans_prob += (trans_prob * v[next_state])
                total_trans_prob *= beta
                rewards.append(env.GetReward(state, action) + total_trans_prob)

            max_reward = max(rewards)
            reward_idx = rewards.index(max_reward)
            arg_max_state, arg_max_action = state_action_pair[reward_idx]

            v_new[state] = max_reward
            pi_new[state] = arg_max_action
            bellman_err = max(bellman_err, abs(v_new[state] - v[state]))

        v = deepcopy(v_new)
        pi = deepcopy(pi_new)
    return v, pi

start_time = time.time()
v, pi = value_iteration(env, beta = 0.99)
v_np, pi_np  = np.array(v), np.array(pi)
print("\nState Value: \n{} \n\nPolicy: \n{}".format(np.array(v_np[:-1]).reshape((map_size,map_size)), np.array(pi_np[:-1]).reshape((map_size,map_size))))
end_time = time.time()
print("Total runtime = ", (end_time - start_time))
