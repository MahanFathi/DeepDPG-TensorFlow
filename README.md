# DeepDPG-TensorFlow
TensorFlow Implementation of [Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1509.02971.pdf)


## Intro 
Replay buffers and target networks, as first proposed in [ATARI playing paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), have made it possible to train deep value networks (DQN) over complicated environments. This is great, but DQN only works fine with discrete domains, since it relies on finding the action that maximizes the action-value function. Insisting on solving continuous valued cases, same authors came up with this model-free off-policy actor-critic algorithm, again by putting the DQN successes to good use. Here the exact algorithm is implemented using TensorFlow for continuous OpenAI Gym environments. 

## Overview
This code contains:
1. Deep Q-Networking and Policy Improvement 
2. Easy Network Setting and [Batch Normalization](https://arxiv.org/pdf/1502.03167.pdf) at Will 
    - changing your network architecture reduces to editing a list
3. Experience Replay Memory 
    - makes the algorithm off-policy 
4. Target Networks for Both Action-Value and Policy Functions
    - stabilizes the learning process
5. It's Modular 

#### A Playground for Controlling OpenAI Gym
Can play around with network setttings in `config.py` and control other environments. 

#### TODOS
- extend it to [MuJoCo](http://www.mujoco.org/) environments 
- saving and loading checkpoints (net weights)
- make nice summaries 

## References
- [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/pdf/1509.02971.pdf)
- [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- [Deterministic Policy Gradient Algorithm](proceedings.mlr.press/v32/silver14.pdf)
- [And this nice repo](https://github.com/devsisters/DQN-tensorflow)
