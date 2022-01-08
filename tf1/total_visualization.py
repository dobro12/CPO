import matplotlib.pyplot as plt
from matplotlib import rc
from copy import deepcopy
import numpy as np
import pickle
import glob
import sys
import os

font = {'size'   : 14}
rc('font', **font)
step_per_episode = 1000

if len(sys.argv) > 1:
    moving_period = int(sys.argv[1])
else:
    moving_period = 10

def main():
    env_name = 'Safexp_PointGoal1'
    '''
    algo_items1 = ["CPO_{}".format(i) for i in [1, 2, 3]]
    algo_items2 = ["PPO_{}".format(i) for i in [1, 2, 3]]
    algo_list = [algo_items1, algo_items2]
    name_list = ["CPO", "PPO"]
    '''
    algo_items1 = ["CPO_{}".format(i) for i in [1, 2, 3]]
    algo_list = [algo_items1]
    name_list = ["CPO"]
    item_list = ['score', 'cost']

    fig_size = 6
    fig, ax_list = plt.subplots(nrows=2, ncols=1, figsize=(fig_size*1.5, fig_size*1.5))

    max_CV = 25
    for idx in range(len(item_list)):
        item_name = item_list[idx]
        ax = ax_list[idx]
        if item_name == 'cost':
            ax.plot([min(lin_space), max(lin_space)], [max_CV, max_CV], 'black', label="constraint")

        for i in range(len(algo_list)):
            algo_items = algo_list[i]
            algo_items = ["{}_{}".format(env_name, algo_item) for algo_item in algo_items]
            lin_space, rewards, stds = draw(algo_items, item_name)
            ax.plot(lin_space, rewards, lw=2, label=name_list[i])
            ax.fill_between(lin_space, rewards-stds, rewards+stds, alpha=0.3)

        ax.set_xlabel('Steps')
        ax.set_ylabel(item_name)
        if item_name == "score":
            ax.set_title('{}\n{}'.format(env_name, "Score"))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        elif item_name == "cost":
            ax.set_title('{}'.format("Constraint Violation"))
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.grid()

    fig.tight_layout()
    plt.savefig('{}_{}.png'.format(env_name, '&'.join(item_list)))
    plt.show()    

def smoothing(records):
    smooth = []
    for i in range(1, len(records)+1):
        if i < moving_period:
            a = np.mean(records[:i])
        else:
            a = np.mean(records[i-moving_period:i])
        smooth.append(a)
    return smooth

def draw(dir_list, item_name):
    dir_names = ['{}/{}_log'.format(dir_item, item_name) for dir_item in dir_list]

    records = []
    for dir_name in dir_names:
        record_names = glob.glob('./{}/*.pkl'.format(dir_name))
        record_names.sort()
        [print(record_name) for record_name in record_names]
        print('-'*10)
        temp_records = []
        for record_name in record_names:
            with open(record_name, 'rb') as f:
                temp_records += pickle.load(f)
        records.append(temp_records)

    steps = []
    rewards = []
    for i in range(len(dir_names)):
        temp_records = records[i]

        temp_step = 0
        temp_steps = [0]
        for step_idx in range(len(temp_records)):
            temp_step += temp_records[step_idx][0]
            temp_steps.append(temp_step)

        temp_rewards = np.array([0] + [record[1] for record in temp_records])
        lin_space = np.linspace(temp_steps[0], temp_steps[-1], int((temp_steps[-1]-temp_steps[0])/step_per_episode))
        temp_rewards = np.interp(lin_space, temp_steps, temp_rewards)
        steps.append(lin_space)
        rewards.append(temp_rewards)

    for i in range(len(dir_names)):
        rewards[i] = smoothing(rewards[i])

    start = max([item[0] for item in steps])
    end = min([item[-1] for item in steps])
    print(start, end)
    lin_space = np.linspace(start, end, 100000)

    for i in range(len(dir_names)):
        rewards[i] = np.interp(lin_space, steps[i], rewards[i])
    stds = np.std(rewards, axis=0)
    rewards = np.mean(rewards, axis=0)
    return lin_space, rewards, stds

if __name__ == "__main__":
    main()
