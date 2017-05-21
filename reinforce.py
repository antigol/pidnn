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
parser.add_argument('--gamma_time', type=float, default=100, metavar='G',
                    help='discount time (default: 100)')
parser.add_argument('--inertia', type=float, default=20, metavar='T')
parser.add_argument('--dissipation', type=float, default=100, metavar='T',
                    help='dissipation')
parser.add_argument('--power', type=float, default=0.1, metavar='P')
parser.add_argument('--int_time', type=float, default=100, metavar='T')
parser.add_argument('--consign_variation', type=float, default=1e-2, metavar='E')
parser.add_argument('--consign_time_step', type=float, default=1000, metavar='T')
parser.add_argument('--learning_rate', type=float, default=0.01, metavar='lr')

args = parser.parse_args()

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.affine1 = nn.Linear(3, 64)
        self.affine2 = nn.Linear(64, 64)
        self.affine3 = nn.Linear(64, 1)

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
    return action.data

def learn():
    if len(policy.rewards) < 200:
        return

    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + exp(-1 / args.gamma_time) * R
        rewards.insert(0, R)

    rewards = rewards[:100]

    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions[:100], rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions[:100], [None for _ in policy.saved_actions[:100]])
    optimizer.step()
    del policy.rewards[:100]
    del policy.saved_actions[:100]

def finish_episode():
    del policy.rewards[:]
    del policy.saved_actions[:]

def main():
    import fake
    import matplotlib.pyplot as plt

    if os.path.isfile('cnn.pkl'):
        policy.load_state_dict(torch.load('cnn.pkl'))

    fig = plt.figure()
    fig.show()

    env = fake.PIDsim(inertia=args.inertia, dissipation=args.dissipation, power=args.power, int_time=args.int_time, consign_variation=args.consign_variation, consign_time_step=args.consign_time_step)

    with InterruptHandler() as h:
        for i_episode in count(0):
            state = env.reset()

            render = 0
            for i in range(100000): # Don't infinite loop while learning
                action = select_action(state)
                state, reward, done = env.step(action[0,0])

                if h.catch():
                    if render > 0:
                        torch.save(policy.state_dict(), 'cnn.pkl')
                        sys.exit(0)
                    render = 5000
                    time = []
                    consigns = []
                    currents = []

                if render > 0:
                    render -= 1
                    time.append(i)
                    consigns.append(env.consign)
                    currents.append(env.current)

                    if i % 10 == 0:
                        fig = plt.figure(fig.number)
                        plt.cla()
                        plt.plot(time, consigns, 'r-')
                        plt.plot(time, currents, 'k-')
                        plt.xlabel('time')
                        plt.ylabel('value')
                        fig.canvas.draw()

                policy.rewards.append(reward)

                learn()

                if done:
                    break

            mean_reward = np.mean(policy.rewards)

            finish_episode()

            print('Episode {} mean reward {:.2}'.format(i_episode, mean_reward))

if __name__ == "__main__":
    main()
