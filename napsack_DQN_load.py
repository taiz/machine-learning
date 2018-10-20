# coding:utf-8

import numpy as np
import random
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

#
# 問題設定
#
CANRIES = [(2,30), (1,2), (3,6), (2,1), (1,3), (5,85)] # (weight,value)
W = 8

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=2):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l1=L.Linear(obs_size, n_hidden_channels)
            self.l2=L.Linear(n_hidden_channels, n_hidden_channels)
            self.l3=L.Linear(n_hidden_channels, n_actions)
    def __call__(self, x, test=False):
        h1 = F.tanh(self.l1(x))
        h2 = F.tanh(self.l2(h1))
        y = chainerrl.action_value.DiscreteActionValue(self.l3(h2))
        return y


def random_action():
    return random.randint(0, len(CANRIES))
 
def step(state, action):
    if action == 0:
        return state, 0, True
    global CANRIES
    if state[action-1] == 0:
        candy = CANRIES[action-1]
        weight, reward = candy
        _state = state.tolist()
        _state[-1] += weight
        if _state[-1] <= W:
            _state[action-1] = 1
            return np.array(_state), reward, False
        else:
            return state, -100, False
    else:
        return state, -100, True

gamma = 0.9
alpha = 0.5
max_number_of_steps = 5
num_episodes = 5000

q_func = QFunction(len(CANRIES) + 1, len(CANRIES) + 1)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes, random_action_func=random_action)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
agent = chainerrl.agents.DQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi
)
agent.load('agent')

state = np.zeros(len(CANRIES) + 1, dtype = int)
for i in range(5):
    action = agent.act(state)
    print(state, action)
    state, _, _ = step(state, action)
