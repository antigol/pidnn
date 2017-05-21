#pylint: disable=R,C,E1101
import numpy as np
from math import exp

class PIDsim:
    def __init__(self, inertia, dissipation, power, int_time, consign_variation):
        self.inertia = inertia
        self.dissipation = dissipation
        self.power = power
        self.int_time = int_time
        self.consign_variation = consign_variation

        self.current = 0
        self.consign = 0
        self.integral = 0
        self.lasterror = 0
        self.instant = 0

    def reset(self):
        self.instant = self.current = np.random.uniform(10)
        self.consign = np.random.uniform(10)
        self.lasterror = self.current - self.consign
        self.integral = self.lasterror

        return np.array([self.lasterror, self.integral, 0])

    def step(self, action):
        self.step_heater(action)
        self.step_consign()

        error = self.current - self.consign
        self.integral = (1 - 1 / self.int_time) * self.integral + error / self.int_time
        differential = error - self.lasterror
        self.lasterror = error
        state = np.array([error, self.integral, differential])
        state = np.clip(state, -1, 1)

        reward = -abs(self.current - self.consign)
        done = reward < -10
        reward = np.clip(reward, -2, 2)
        return state, reward, done

    def step_heater(self, action):
        self.instant *= 1 - self.dissipation
        self.instant += self.power * action
        self.current = self.inertia * self.current + (1 - self.inertia) * self.instant

    def step_consign(self):
        self.consign += self.consign_variation * np.random.normal()
        if self.consign < 0:
            self.consign = 0
