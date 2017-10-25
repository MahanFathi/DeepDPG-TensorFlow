import gym
import numpy as np


class Environment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)
        # Make sure it is not a discrete env
        assert not isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.shape[0]

    def reset(self):
        s = self.env.reset()
        s = np.reshape(s, [1, -1])
        return s

    def act(self, action, animate):
        s2, r, terminal, _ = self.env.step(action)
        s2 = np.reshape(s2, [1, -1])
        if animate:
            self.env.render()
        return s2, r, terminal
