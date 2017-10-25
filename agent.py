import tensorflow as tf
import numpy as np
from netclass import Nets
from exp_replay import BufferMemo


class Agent(Nets):
    def __init__(self, config, environment, sess):
        self.config = config
        self.sess = sess
        self.env = environment
        super(Agent, self).__init__(config, environment, sess)
        self.memory = BufferMemo(self.config, self.env)
        self.noise = OrnsteinUhlenbeckActionNoise(self.config, self.env)
        self.step = 0
        self.mean_return = 0

    def train(self):
        for ep in range(self.config.episode_num):
            if ep is not 0 and ep % self.config.test_every == 0 and self.config.test:
                self.test()
            s1 = self.env.reset()
            ep_return = 0
            self.noise.reset()
            for t in range(self.config.episode_maxlength):
                a = self.sample_action(s1)
                s2, r, terminal = self.env.act(a, self.config.animate_training)
                self.memory.add(s1, a, s2, r, terminal)
                if self.memory.ready:
                    state1s, actions, state2s, rewards, terminals = self.memory.sample()
                    self.train_critic(state1s, actions, state2s, rewards, terminals)
                    self.train_actor(state1s)
                    self.update_actor_target()
                    self.update_critic_target()
                s1 = s2
                ep_return += r
                if terminal:
                    self.log(ep, ep_return)
                    break

    def test(self):
        for ep in range(self.config.test_num):
            s1 = self.env.reset()
            ep_return = 0
            for t in range(self.config.episode_maxlength):
                a = self.sample_action(s1, is_testing=True)
                s2, r, terminal = self.env.act(a, self.config.animate_testing)
                s1 = s2
                ep_return += r
                if terminal:
                    self.log(ep, ep_return, is_testing=True)
                    break
        if self.config.animate_testing and not self.config.animate_training:
            self.env.env.render(close=True)

    def sample_action(self, obs, is_testing=False):
        if is_testing:
            return self.sample_mu_action(obs)
        else:
            return self.sample_mu_action(obs) + self.noise()

    def log(self, ep, ep_return, is_testing=False):
        self.mean_return += ep_return
        if is_testing:
            print('     | :Test Time: | Episode: {:d} | Reward: {:d} |'.format(ep, int(ep_return)))
        else:
            print('| Episode: {:d} | Reward: {:d} |'.format(ep, int(ep_return)))


class OrnsteinUhlenbeckActionNoise:
    """
    Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
    based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """

    def __init__(self, config, environment):
        self.theta = config.theta
        self.mu = np.zeros([1, environment.ac_dim])
        self.sigma = config.sigma
        self.dt = config.dt
        self.x0 = config.x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return np.float32(x)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
