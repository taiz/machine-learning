# coding:utf-8

import numpy as np
import random
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import time

#
# 問題設定
#
#CANDIES = [(2,2), (1,2), (3,6), (2,1), (1,3), (5,85)] # (weight,value)
CANDIES = [(2,2), (1,2), (3,6), (2,1), (1,3), (5,85), (1,0), (1,0), (1,0), (1,0), (1,0)] # (weight,value)
N = len(CANDIES)
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

global_state = None

def random_action():
    ret =  random.randint(0, N)
    return ret
    #weight = global_state[0]
    #indexies = [0]
    #for i in range(len(global_state)-1):
    #    if global_state[i+1] == 0 and CANDIES[i][0] + weight <= W:
    #        indexies.append(i+1)
    #ret = np.random.choice(indexies)
    #return ret
 
def step(state, action):
    if action == 0:
        return state, 0, True
    global CANDIES
    if state[action] == 0:
        candy = CANDIES[action-1]
        weight, reward = candy
        _state = state.tolist()
        _state[0] += weight
        if _state[0] <= W:
            _state[action] = 1
            return np.array(_state), reward, False
        else:
            return state, -100, True
    else:
        return state, -100, True

gamma = 0.9
alpha = 0.5
max_number_of_steps = 5
num_episodes = 5000

q_func = QFunction(N + 1, N + 1, 32 * N)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(start_epsilon=1.0, end_epsilon=0.1, decay_steps=num_episodes, random_action_func=random_action)
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
phi = lambda x: x.astype(np.float32, copy=False)
agent = chainerrl.agents.DQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1, target_update_interval=100, phi=phi
)

start = time.time()

for episode in range(num_episodes):
    global_state = state = np.zeros(1 + N, dtype = int)
    R = 0
    reward = 0
    done = False
 
    for t in range(max_number_of_steps):
        action = agent.act_and_train(state, reward)
        next_state, reward, done = step(state, action)
        global_state = state = next_state
        R += reward
        print(state, action, reward, R)
        #if done:
        #    break
    agent.stop_episode_and_train(state, reward, done)

    print('episode:', episode, 'R:', R+1, 'statistics:', agent.get_statistics())

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

agent.save('agent')

print("--------------------")
state = np.zeros(N + 1, dtype = int)
reward = 0
for i in range(5):
    action = agent.act(state)
    state, _reward, _ = step(state, action)
    reward += _reward
    print(state, action, reward)
