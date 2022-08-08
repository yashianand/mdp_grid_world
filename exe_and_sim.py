from grid_world import gridWorld, printEnvironment
from policy_iteration import policyIteration
from value_iteration import valueIteration
import numpy as np

def play_episodes(grid, trials, policy):
        rewards = []
        for _ in range(trials):
            terminated = False
            grid.reset()
            print("here: ", grid.state, policy[0])
            transitions = grid.step(int(policy[0]))


v_VI, policy_VI = valueIteration(gridWorld, gamma=0.99)
play_episodes(gridWorld, trials=1, policy=policy_VI)
# v_PI, policy_PI = policyIteration(gridWorld, gamma=0.99)
