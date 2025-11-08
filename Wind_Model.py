import numpy as np
import random
import matplotlib.pyplot as plt
import math
import atexit
import signal
import sys
import os


class Wind_Model:
    """Генератор ветровых возмущений.
    """
    def __init__(self, dt,
                 steady=np.zeros(3),
                 gm_sigma=np.zeros(3), gm_tau=np.ones(3),
                 gust_events=None,
                 pulse_rate=0.0, pulse_amp_range=(0.0, 0.0), pulse_dur_range=(0.0, 0.0),
                 seed=None):
        self.dt = dt
        self.steady = np.array(steady, dtype=float)
        self.gm_sigma = np.array(gm_sigma, dtype=float)
        self.gm_tau = np.array(gm_tau, dtype=float)
        self.state_gm = np.zeros(3)
        self.gust_events = gust_events or []
        self.pulse_rate = pulse_rate
        self.pulse_amp_range = pulse_amp_range
        self.pulse_dur_range = pulse_dur_range
        self.active_pulses = []
        self.rng = np.random.default_rng(seed)

    # --------------------
    # >>> ВЕТЕР: СТУПЕНЬ — дискретный порыв (step gust)
    # --------------------
    @staticmethod
    def step_gust(t, t0, T, A):
        if t < t0 or t > t0 + T:
            return np.zeros(3)
        return np.array(A, dtype=float)

    def _update_gauss_markov(self):
        if np.all(self.gm_sigma == 0):
            return np.zeros(3)
        coef = np.exp(-self.dt / self.gm_tau)
        noise_scale = self.gm_sigma * np.sqrt(1 - coef**2)
        self.state_gm = coef * self.state_gm + noise_scale * self.rng.standard_normal(3)
        return self.state_gm

    def _update_random_pulses(self, t):
        # очистка просроченных импульсов
        self.active_pulses = [p for p in self.active_pulses if t < p['t_end']]
        # генерация нового импульса
        if self.pulse_rate > 0 and self.rng.random() < self.pulse_rate * self.dt:
            amp = self.rng.uniform(*self.pulse_amp_range)
            dur = self.rng.uniform(*self.pulse_dur_range)
            direction = self.rng.normal(size=3)
            direction /= np.linalg.norm(direction) + 1e-9
            vec = amp * direction
            self.active_pulses.append({'vec': vec, 't_end': t + dur})
        if not self.active_pulses:
            return np.zeros(3)
        return np.sum([p['vec'] for p in self.active_pulses], axis=0)

    def step(self, t):
        v = self.steady.copy()
        v += self._update_gauss_markov()
        for g in self.gust_events:
            v += self.step_gust(t, g['t_start'], g['duration'], g['amplitude'])
        v += self._update_random_pulses(t)
        return v