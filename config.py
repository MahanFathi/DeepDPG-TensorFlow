import tensorflow as tf
import numpy as np


class NetConfig(object):
    env_name = 'Pendulum-v0'
    # env_name = 'MountainCarContinuous-v0'

    animate_training = True

    test = False
    animate_testing = True
    test_every = 10
    test_num = 5

    actor_net_size = [(128, True), (64, True)]
    actor_net_activation_fn = tf.nn.relu

    critic_net_size = [(128, True), (64, False)]
    critic_net_ac_size = []
    critic_net_activation_fn = tf.nn.relu
    critic_net_junction = 1

    critic_learning_rate = 5e-3
    actor_learning_rate = 1e-3

    gamma = 0.99
    tau = 1e-3

    buffer_size = 100000
    batch_size = 64
    episode_num = 100
    episode_maxlength = 1000

    # OrnsteinUhlenbeckActionNoise Properties
    sigma = 0.3
    theta = .15
    dt = 1e-2
    x0 = None

    # You gonna get nice results after ~50 episodes by this setup
