#pylint: disable=R,C,E1101
import argparse
from math import exp
import numpy as np
from itertools import count
import signal
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class InterruptHandler:

    def __init__(self, sig=signal.SIGINT):
        self.sig = sig
        self.interrupted = False
        self.original_handler = None

    def __enter__(self):
        self.original_handler = signal.getsignal(self.sig)

        def handler(signum, frame):
            self.interrupted = True
        signal.signal(self.sig, handler)
        return self

    def __exit__(self, type, value, tb):
        signal.signal(self.sig, self.original_handler)

    def catch(self):
        if self.interrupted:
            self.interrupted = False
            return True
        return False

parser = argparse.ArgumentParser(description='PIDNN PID by neural network')
parser.add_argument('--gamma_time', type=float, default=200, metavar='T')
parser.add_argument('--inertia', type=float, default=20, metavar='T')
parser.add_argument('--dissipation', type=float, default=100, metavar='T')
parser.add_argument('--int_time', type=float, default=200, metavar='T')
parser.add_argument('--power', type=float, default=0.1, metavar='P')
parser.add_argument('--consign_variation', type=float, default=1e-2, metavar='E')
parser.add_argument('--consign_time_step', type=float, default=1000, metavar='T')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='lr')
parser.add_argument('--render_time', type=int, default=1000, metavar='T')
parser.add_argument('--memory_time', type=int, default=10, metavar='T')

args = parser.parse_args()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(4, 16)
        self.affine2 = nn.Linear(16, 16)
        self.affine3 = nn.Linear(16, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x): #pylint: disable=W
        x = self.affine1(x)
        x = F.sigmoid(x)
        x = self.affine2(x)
        x = F.sigmoid(x)
        x = self.affine3(x)
        return F.sigmoid(x)

policy = Policy()
policy.train()
optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)

def select_action(state):
    state = np.clip(state, -1, 1)
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = probs.bernoulli()
    policy.saved_actions.append(action)
    return action.data[0, 0]

def learn():
    n = 100
    k = 1000
    if len(policy.rewards) < n + k:
        return

    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + exp(-1 / args.gamma_time) * R
        rewards.insert(0, R)

    rewards = rewards[:n]

    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions[:n], rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions[:n], [None for _ in policy.saved_actions[:n]])
    optimizer.step()
    del policy.rewards[:n]
    del policy.saved_actions[:n]

def finish_episode():
    del policy.rewards[:]
    del policy.saved_actions[:]

def main():
    import fake
    import matplotlib.pyplot as plt
    import IPython

    if os.path.isfile('cnn.pkl'):
        policy.load_state_dict(torch.load('cnn.pkl'))
        IPython.embed()

    fig = plt.figure()
    fig.show()

    env = fake.PIDsim(inertia=args.inertia, dissipation=args.dissipation, power=args.power, int_time=args.int_time, consign_variation=args.consign_variation, consign_time_step=args.consign_time_step)

    with InterruptHandler() as h:
        for i_episode in count(0):
            state = env.reset()

            memory = 0
            state = np.concatenate((state, [memory]))

            render = 0
            for i in range(50000): # Don't infinite loop while learning
                action = select_action(state)
                memory = (1 - exp(-1 / args.memory_time)) * memory + exp(-1 / args.memory_time) * action
                state, reward, done = env.step(action)
                state = np.concatenate((state, [memory]))

                if h.catch():
                    if render > 0:
                        torch.save(policy.state_dict(), 'cnn.pkl')
                        sys.exit(0)
                    render = args.render_time
                    time = []
                    consigns = []
                    currents = []
                    powers = []

                if render > 0:
                    render -= 1
                    time.append(i)
                    consigns.append(env.consign)
                    currents.append(env.current)
                    powers.append(action)

                    if i % 5 == 0:
                        fig = plt.figure(fig.number)
                        plt.cla()
                        plt.plot(time, consigns, 'r-')
                        plt.plot(time, currents, 'k-')
                        plt.plot(time, powers, 'b-')
                        plt.xlabel('time')
                        plt.ylabel('value')
                        fig.canvas.draw()

                policy.rewards.append(reward)

                if i % 1000 == 0:
                    print('Episode {}/{} mean reward {:.2}'.format(i_episode, i, np.mean(policy.rewards)))
                learn()

                if done:
                    break

            finish_episode()

if __name__ == "__main__":
    main()
