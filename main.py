import numpy as np
import tensorflow as tf
from agent import Agent
from environment import Environment
from config import NetConfig


# EDIT CONFIG AND RUN THIS

def main():
    config = NetConfig
    sess = tf.Session()
    environment = Environment(config)
    agent = Agent(config, environment, sess)
    agent.train()


if __name__ == '__main__':
    main()
