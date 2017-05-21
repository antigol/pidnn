#pylint: disable=R,C,E1101
import numpy as np
from math import exp

class PIDsim:
    def __init__(self, inertia, dissipation, power, int_time, consign_variation, consign_time_step):
        self.inertia = exp(-1 / inertia)
        self.dissipation = exp(-1 / dissipation)
        self.power = power
        self.int_time = exp(-1 / int_time)
        self.consign_variation = consign_variation
        self.consign_pstep = 1 / consign_time_step

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
        self.integral = self.int_time * self.integral + (1 - self.int_time) * error
        differential = error - self.lasterror
        self.lasterror = error
        state = np.array([error, self.integral, differential])

        reward = -abs(self.current - self.consign)
        done = reward < -10
        # reward = np.clip(reward, -1, 1)
        return state, reward, done

    def step_heater(self, action):
        self.instant *= self.dissipation
        self.instant += self.power * action
        self.current = self.inertia * self.current + (1 - self.inertia) * self.instant

    def step_consign(self):
        if np.random.rand() < self.consign_pstep:
            self.consign += self.consign_variation * np.random.normal()
        self.consign = np.clip(self.consign, 0, self.power / (1 - self.dissipation))
