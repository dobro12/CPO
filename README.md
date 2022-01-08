# CPO
A simple Tensorflow1 & PyTorch implementation of constrained policy optimization (CPO) on Safety Gym.

## requirement
- gym
- mujoco
- safety_gym
- tensorflow 1.13.1
- pytorch 1.10.1

## how to use

### tf1
- `python train.py #training`
- `python train.py test #test`

### torch
- `python main.py #training`
- `python main.py --test --resume {#_of_checkpoint} #test`

## reference
- CPO paper: https://arxiv.org/abs/1705.10528
- Original code: https://github.com/openai/safety-starter-agents
