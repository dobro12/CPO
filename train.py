# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################

from graph_drawer import Graph
from logger import Logger
from nets import Agent

from collections import deque
import tensorflow as tf
import numpy as np
import safety_gym
import random
import pickle
import time
import sys
import gym

#for random seed
seed = 2
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

env_name = 'Safexp-PointGoal1-v0'
#env_name = 'Safexp-CarGoal1-v0'

agent_name = 'CPO'
algo = '{}_{}'.format(agent_name, seed)
save_name = '_'.join(env_name.split('-')[:-1])
save_name = "{}_{}".format(save_name, algo)
agent_args = {'agent_name':agent_name,
            'env_name':save_name,
            'discount_factor':0.99,
            'hidden1':256,
            'hidden2':256,
            'v_lr':1e-3,
            'cost_v_lr':1e-3,
            'value_epochs':80,
            'cost_value_epochs':80,
            'num_conjugate':10,
            'max_decay_num':10,
            'line_decay':0.8,
            'max_kl':0.01,
            'max_avg_cost':25/1000,
            'damping_coeff':0.01,
            'gae_coeff':0.97,
            }

def hazard_dist(hazard_pos_list, pos):
    pos = np.array(pos)
    min_dist = np.inf
    for hazard_pos in hazard_pos_list:
        dist = np.sqrt(np.sum(np.square(hazard_pos[:2] - pos[:2])))
        if dist < min_dist:
            min_dist = dist
    return min_dist

def get_cost(dist, h_size=0.2, h_coeff=10.0):
    return 1/(1+np.exp((dist - h_size)*h_coeff))

def train():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    v_loss_logger = Logger(save_name, 'v_loss')
    cost_v_loss_logger = Logger(save_name, 'cost_v_loss')
    kl_logger = Logger(save_name, 'kl')
    score_logger = Logger(save_name, 'score')
    cost_logger = Logger(save_name, 'cost')
    graph = Graph(1000, save_name, ['score', 'cost', 'value loss', 'cost value loss', 'kl divergence'])
    max_steps = 4000
    max_ep_len = 1000
    episodes = int(max_steps/max_ep_len)
    epochs = 500
    save_freq = 10

    log_length = 10
    p_objectives = deque(maxlen=log_length)
    c_objectives = deque(maxlen=log_length)
    v_losses = deque(maxlen=log_length)
    cost_v_losses = deque(maxlen=log_length)
    kl_divergence = deque(maxlen=log_length)
    scores = deque(maxlen=log_length*episodes)
    costs = deque(maxlen=log_length*episodes)

    for epoch in range(epochs):
        states = []
        actions = []
        targets = []
        cost_targets = []
        gaes = []
        cost_gaes = []
        avg_costs = []
        ep_step = 0
        while ep_step < max_steps:
            state = env.reset()
            done = False
            score = 0
            cost = 0
            step = 0
            temp_rewards = []
            temp_costs = []
            values = []
            cost_values = []
            while True:
                step += 1
                ep_step += 1
                assert env.observation_space.contains(state)
                action, clipped_action, value, cost_value = agent.get_action(state, True)
                assert env.action_space.contains(clipped_action)
                next_state, reward, done, info = env.step(clipped_action)

                #for predict cost
                h_dist = hazard_dist(env.hazards_pos, env.world.robot_pos())
                predict_cost = get_cost(h_dist)

                states.append(state)
                actions.append(action)
                temp_rewards.append(reward)
                temp_costs.append(predict_cost)
                values.append(value)
                cost_values.append(cost_value)

                state = next_state
                score += reward
                cost += info.get('cost', 0) #로그는 실제 cost를 남겨서, discrete한 cost랑 비교해야함.

                if done or step >= max_ep_len:
                    break

            if step >= max_ep_len:
                action, clipped_action, value, cost_value = agent.get_action(state, True)
            else:
                value = 0
                cost_value = 0
                print("done before max_ep_len...") 
            next_values = values[1:] + [value]
            temp_gaes, temp_targets = agent.get_gaes_targets(temp_rewards, values, next_values)
            next_cost_values = cost_values[1:] + [cost_value]
            temp_cost_gaes, temp_cost_targets = agent.get_gaes_targets(temp_costs, cost_values, next_cost_values)
            avg_costs.append(np.mean(temp_costs))
            targets += list(temp_targets)
            gaes += list(temp_gaes)
            cost_targets += list(temp_cost_targets)
            cost_gaes += list(temp_cost_gaes)

            score_logger.write([step, score])
            cost_logger.write([step, cost])
            scores.append(score)
            costs.append(cost)

        trajs = [states, actions, targets, cost_targets, gaes, cost_gaes, avg_costs]
        v_loss, cost_v_loss, p_objective, cost_objective, kl = agent.train(trajs)

        v_loss_logger.write([ep_step, v_loss])
        cost_v_loss_logger.write([ep_step, cost_v_loss])
        kl_logger.write([ep_step, kl])

        p_objectives.append(p_objective)
        c_objectives.append(cost_objective)
        v_losses.append(v_loss)
        cost_v_losses.append(cost_v_loss)
        kl_divergence.append(kl)

        print(np.mean(scores), np.mean(costs), np.mean(v_losses), np.mean(cost_v_losses), np.mean(kl_divergence), np.mean(c_objectives))
        graph.update([np.mean(scores), np.mean(costs), np.mean(v_losses), np.mean(cost_v_losses), np.mean(kl_divergence)])
        if (epoch+1)%save_freq == 0:
            agent.save()
            v_loss_logger.save()
            cost_v_loss_logger.save()
            kl_logger.save()
            score_logger.save()
            cost_logger.save()

    graph.update(None, finished=True)

def test():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    episodes = int(1e6)

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, clipped_action, value, cost_value = agent.get_action(state, False)
            #action, clipped_action, value, cost_value = agent.get_action(state, True)
            state, reward, done, info = env.step(clipped_action)
            print(reward, '\t', info.get('cost', 0))
            score += reward
            env.render()
        print("score :",score)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()
