import numpy as np
import tensorflow as tf
from buildnet import build_net_1, build_net_2


class Nets(object):
    def __init__(self, config, environment, sess):
        self.actor_net_size = config.actor_net_size
        self.actor_net_activation_fn = config.actor_net_activation_fn
        self.critic_net_size = config.critic_net_size
        self.critic_net_ac_size = config.critic_net_ac_size
        self.critic_net_activation_fn = config.critic_net_activation_fn
        self.critic_net_junction = config.critic_net_junction
        self.critic_learning_rate = config.critic_learning_rate
        self.actor_learning_rate = config.actor_learning_rate
        self.gamma = config.gamma
        self.tau = config.tau
        self.ob_dim = environment.ob_dim
        self.ac_dim = environment.ac_dim
        self.max_action = environment.env.action_space.high
        self.sess = sess
        self.sess.__enter__()
        self.build_nets()

    def build_nets(self):
        # Define Placeholders
        self.sy_ob_no = tf.placeholder(shape=[None, self.ob_dim], name='observation', dtype=tf.float32)
        self.sy_ac_na = tf.placeholder(shape=[None, self.ac_dim], name='action', dtype=tf.float32)
        self.predicted_q = tf.placeholder(shape=[None, 1], name='predicted_q', dtype=tf.float32)
        # Build Nets
        self.sy_mu_na, self.w_mu = build_net_1(self.sy_ob_no, self.ac_dim,
                                               scope='Actor_Net', size=self.actor_net_size,
                                               activation_fn=self.actor_net_activation_fn,
                                               output_activation=tf.tanh)
        self.sy_mu_na = tf.multiply(self.sy_mu_na, self.max_action)
        self.w_mu_as_list = tf.trainable_variables()
        self.sy_q_n, self.w_q = build_net_2(self.sy_ob_no, self.sy_ac_na,
                                            1, junction=self.critic_net_junction,
                                            scope='Q_Net', size=self.critic_net_size,
                                            ac_size=self.critic_net_ac_size, activation_fn=self.critic_net_activation_fn)
        # Build Target Nets
        self.sy_tar_mu_na, self.w_tar_mu = build_net_1(self.sy_ob_no, self.ac_dim,
                                                       scope='Target_Actor_Net', size=self.actor_net_size,
                                                       activation_fn=self.actor_net_activation_fn,
                                                       output_activation=tf.tanh)
        self.sy_tar_mu_na = tf.multiply(self.sy_tar_mu_na, self.max_action)
        self.sy_tar_q_n, self.w_tar_q = build_net_2(self.sy_ob_no, self.sy_ac_na,
                                                    1, junction=self.critic_net_junction,
                                                    scope='Target_Q_Net', size=self.critic_net_size,
                                                    ac_size=self.critic_net_ac_size, activation_fn=self.critic_net_activation_fn)


        # Define Optimization Process
        # Critic (predicted_q --> r + gamma * Q')
        self.loss_critic = tf.reduce_mean(tf.square(self.predicted_q - self.sy_q_n))
        self.optimize_critic = tf.train.AdamOptimizer(self.critic_learning_rate).minimize(self.loss_critic)
        # Actor (chain rule)
        self.action_gradient = tf.gradients(self.sy_q_n, self.sy_ac_na)
        self.sy_action_gradient = tf.placeholder(tf.float32, [None, self.ac_dim])
        self.actor_gradients = tf.gradients(self.sy_mu_na, self.w_mu_as_list, -self.sy_action_gradient)
        self.optimize_actor = tf.train.AdamOptimizer(self.actor_learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.w_mu_as_list))

        # Target Networks Update Ops
        with tf.variable_scope('transfer_weights'):
            self.sy_tau = tf.placeholder(tf.float32, shape=[])
            self.sy_w_mu_assign = {}
            self.sy_w_tar_mu_assign = {}
            self.w_mu_assign_op = {}
            for name in self.w_mu.keys():
                self.sy_w_mu_assign[name] = tf.placeholder('float32', self.w_mu[name].get_shape().as_list(), name=name)
                self.sy_w_tar_mu_assign[name] = tf.placeholder('float32', self.w_tar_mu[name].get_shape().as_list(), name=name)
                self.w_mu_assign_op[name] = tf.assign(self.w_tar_mu[name],
                                                      self.sy_tau * self.sy_w_mu_assign[name] + (1. - self.sy_tau) * self.sy_w_tar_mu_assign[name])
            self.sy_w_q_assign = {}
            self.sy_w_tar_q_assign = {}
            self.w_q_assign_op = {}
            for name in self.w_q.keys():
                self.sy_w_q_assign[name] = tf.placeholder('float32', self.w_q[name].get_shape().as_list(), name=name)
                self.sy_w_tar_q_assign[name] = tf.placeholder('float32', self.w_tar_q[name].get_shape().as_list(), name=name)
                self.w_q_assign_op[name] = tf.assign(self.w_tar_q[name],
                                                     self.sy_tau * self.sy_w_q_assign[name] + (1. - self.sy_tau) * self.sy_w_tar_q_assign[name])

        tf.global_variables_initializer().run()
        # Make sure training net and target net weights are initialized to the same values
        for name in self.w_mu.keys():
            self.w_mu_assign_op[name].eval({
                self.sy_w_mu_assign[name]: self.w_mu[name].eval(),
                self.sy_w_tar_mu_assign[name]: self.w_tar_mu[name].eval(),
                self.sy_tau: 1.
            })
        for name in self.w_q.keys():
            self.w_q_assign_op[name].eval({
                self.sy_w_q_assign[name]: self.w_q[name].eval(),
                self.sy_w_tar_q_assign[name]: self.w_tar_q[name].eval(),
                self.sy_tau: 1.
            })

        # Sanity Check
        # for name in self.w_tar_mu.keys():
        #     assert self.w_mu[name].eval().all() == self.w_tar_mu[name].eval().all()
        # for name in self.w_tar_q.keys():
        #     assert self.w_q[name].eval().all() == self.w_tar_q[name].eval().all()


    def train_actor(self, obs):
        actor_actions = self.sess.run(self.sy_mu_na, feed_dict={
            self.sy_ob_no: obs
        })
        action_gradient = self.sess.run(self.action_gradient, feed_dict={
            self.sy_ob_no: obs,
            self.sy_ac_na: actor_actions
        })
        self.sess.run(self.optimize_actor, feed_dict={
            self.sy_ob_no: obs,
            self.sy_action_gradient: action_gradient[0]
        })

    def train_critic(self, obs1, acs, obs2, rewards, terminals):
        mu_action = self.sy_tar_mu_na.eval({
            self.sy_ob_no: obs2
        })
        pred_q = self.sess.run(self.sy_tar_q_n, feed_dict={
            self.sy_ob_no: obs2,
            self.sy_ac_na: mu_action
        })
        pred_q *= self.gamma * (1 - terminals)
        pred_q += rewards
        self.sess.run(self.optimize_critic, feed_dict={
            self.sy_ob_no: obs1,
            self.sy_ac_na: acs,
            self.predicted_q: pred_q
        })

    def update_actor_target(self, tau=None):
        if tau is None:
            tau = self.tau
        for name in self.w_mu.keys():
            self.w_mu_assign_op[name].eval({
                self.sy_w_mu_assign[name]: self.w_mu[name].eval(),
                self.sy_w_tar_mu_assign[name]: self.w_tar_mu[name].eval(),
                self.sy_tau: tau
            })

    def update_critic_target(self, tau=None):
        if tau is None:
            tau = self.tau
        for name in self.w_q.keys():
            self.w_q_assign_op[name].eval({
                self.sy_w_q_assign[name]: self.w_q[name].eval(),
                self.sy_w_tar_q_assign[name]: self.w_tar_q[name].eval(),
                self.sy_tau: tau
            })

    def sample_mu_action(self, obs):
        actions = self.sess.run(self.sy_mu_na, feed_dict={
            self.sy_ob_no: obs
        })
        return actions

