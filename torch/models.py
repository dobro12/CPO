from torch.nn import functional
from torch import jit, nn
import torch

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8

def initWeights(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)

def initWeights2(m):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(-1.0, 0.01)

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()

        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']

        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.act_fn = torch.relu
        self.output_act_fn = torch.sigmoid

        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        self.fc_log_std = nn.Linear(self.hidden2_units, self.action_dim)


    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        mean = self.output_act_fn(self.fc_mean(x))
        log_std = self.fc_log_std(x)

        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, log_std, std

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx != 3:
                module.apply(initWeights)
            else:
                module.apply(initWeights2)


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()

        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.hidden1_units = args['hidden1']
        self.hidden2_units = args['hidden2']

        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        self.act_fn = torch.relu


    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.act_fn(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1,))
        return x

    def initialize(self):
        self.apply(initWeights)
