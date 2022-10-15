from grid_world_mdp import gridWorld, printEnvironment
from policy_iteration import policyIteration
from value_iteration import valueIteration
import numpy as np

def play_episodes(grid, trials, policy):
    rewards = []
    for _ in range(trials):
        trial_reward = 0
        terminal= False
        grid.reset()
        observation, reward, prob, terminal = grid.step(int(policy[0]))
        trial_reward += reward
        while not terminal:
            observation, reward, prob, terminal = grid.step(int(policy[observation]))
            trial_reward += reward
        rewards.append(trial_reward)
    return np.average(rewards), np.std(rewards)

print("\nValue Iteration:")
v_VI, policy_VI = valueIteration(gridWorld, gamma=0.99)
print("\nFinal State Values:")
printEnvironment(np.array(v_VI[:]).reshape((4,4)))
print("\nFinal Policy:")
printEnvironment(np.array(policy_VI[:], dtype=int).reshape((4,4)), policy=True)

avg_r_VI, std_dev_VI = play_episodes(gridWorld, trials=10000, policy=policy_VI)
print("Average Reward: {}, Standard Deviation: {}".format(avg_r_VI, std_dev_VI))

# print("\nPolicy Iteration:")
# v_PI, policy_PI = policyIteration(gridWorld, gamma=0.99)
# print("\nFinal State Values:")
# printEnvironment(np.array(v_PI[:]).reshape((4,4)))
# print("\nFinal Policy:")
# printEnvironment(np.array(policy_PI[:], dtype=int).reshape((4,4)), policy=True)

# avg_r_PI, std_dev_PI = play_episodes(gridWorld, trials=10000, policy=policy_PI)
# print("Average Reward: {}, Standard Deviation: {}".format(avg_r_PI, std_dev_PI))
