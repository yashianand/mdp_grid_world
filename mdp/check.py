import numpy as np

class LRTDP:
    def __init__(self, env, discount_factor=0.99, threshold=1e-3):
        self.env = env
        self.discount_factor = discount_factor
        self.threshold = threshold
        self.values = {state: 0 for state in env.states}
        self.solved_states = set()

    def bellman_backup(self, state):
        if state in self.env.obstacles or self.env.is_terminal(state):
            return self.values[state]

        v = float('-inf')
        for action in self.env.actions:
            next_state = self.env.transition(state, action)
            reward = self.env.reward(state, action, next_state)
            v = max(v, reward + self.discount_factor * self.values[next_state])
        self.values[state] = v
        return v

    def check_solved(self, state):
        for action in self.env.actions:
            next_state = self.env.transition(state, action)
            if abs(self.values[state] - (self.env.reward(state, action, next_state) + self.discount_factor * self.values[next_state])) > self.threshold:
                return False
        self.solved_states.add(state)
        return True

    def trial(self, state):
        visited = set()
        stack = [state]
        while stack:
            curr_state = stack[-1]
            if curr_state in self.solved_states:
                stack.pop()
                continue

            visited.add(curr_state)
            if self.env.is_terminal(curr_state):
                self.values[curr_state] = 0
                self.solved_states.add(curr_state)
                stack.pop()
                continue

            v_before = self.values[curr_state]
            self.bellman_backup(curr_state)
            if abs(self.values[curr_state] - v_before) < self.threshold:
                self.solved_states.add(curr_state)
                stack.pop()
                continue

            # Get best action from current state and push its successor onto the stack
            best_action_value = float('-inf')
            best_action = None
            for action in self.env.actions:
                next_state = self.env.transition(curr_state, action)
                q_value = self.env.reward(curr_state, action, next_state) + self.discount_factor * self.values[next_state]
                if q_value > best_action_value:
                    best_action_value = q_value
                    best_action = action

            next_state = self.env.transition(curr_state, best_action)
            if next_state not in visited and next_state not in self.solved_states:
                stack.append(next_state)

    def solve(self, start_state):
        while start_state not in self.solved_states:
            self.trial(start_state)
        return self.values


lrtdp(sspWorld, sspWorld.state, epsilon=0.1)

lrtdp = LRTDP(env)
values = lrtdp.solve(start)
print(values)
