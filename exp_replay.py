import numpy as np


# Tiny class to handle the experience replay
class BufferMemo(object):
    def __init__(self, config, environment):
        self.memo_size = config.buffer_size
        self.state1s = np.empty((self.memo_size, environment.ob_dim), dtype=np.float32)
        self.state2s = np.empty((self.memo_size, environment.ob_dim), dtype=np.float32)
        self.actions = np.empty((self.memo_size, environment.ac_dim), dtype=np.float32)
        self.rewards = np.empty(self.memo_size, dtype=np.float32)
        self.terminals = np.empty(self.memo_size, dtype=np.bool)
        self.count = 0
        self.current = 0
        self.batch_size = config.batch_size
        self.ready_full = False
        self.ready_actor = False
        self.ready_critic = False

    def add(self, state1, action, state2, reward, terminal):
        self.state1s[self.current] = state1
        self.actions[self.current] = action
        self.state2s[self.current] = state2
        self.rewards[self.current] = reward
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memo_size
        self.ready = self.count >= self.batch_size
        self.ready_full = self.count >= self.memo_size

    def sample(self):
        if self.ready_full:
            inds = np.random.choice(self.memo_size, self.batch_size, replace=False)
        else:
            inds = np.random.choice(self.count, self.batch_size, replace=False)
        return self.state1s[inds], self.actions[inds], self.state2s[inds], self.rewards[inds].reshape([-1, 1]), self.terminals[inds].reshape([-1, 1])
