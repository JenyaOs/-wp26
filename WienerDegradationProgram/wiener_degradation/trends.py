"""Классы трендов для винеровской деградационной модели с аналитическими производными."""

import numpy as np
from abc import ABC, abstractmethod


class Trend(ABC):
    """Базовый класс для тренда ρ(t)."""

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def func(self, x, t, c=0):
        """Вычисление значения тренда ρ(t)."""
        pass

    # Методы для производных (по умолчанию 0, переопределяются в наследниках)
    def d_func_gamma(self, t, x, c=0): return 0.0
    def d_func_beta(self, t, x, c=0): return 0.0
    def d2_func_gamma(self, t, x, c=0): return 0.0
    def d2_func_beta(self, t, x, c=0): return 0.0
    def d2_func_gamma_beta(self, t, x, c=0): return 0.0


class LinearTrend(Trend):
    """Линейный тренд: ρ(t) = t"""
    name = "linear"
    def func(self, x, t, c=0): return t

    def invFunc(self, x, z0, c=0): return z0


class PowerTrend(Trend):
    """Степенной тренд: ρ(t) = t^γ"""
    name = "power"
    def func(self, x, t, c=0): return np.power(t, x[2])

    def invFunc(self, x, z0, c=0): return z0 ** (1 / x[2])

    def d_func_gamma(self, t, x, c=0):
        if (t==0):
            return 0
        return np.power(t, x[2]) * np.log(t)

    def d2_func_gamma(self, t, x, c=0):
        return np.power(t, x[2]) * (np.log(t) ** 2)


class LinearWithCovariateTrend(Trend):
    """Линейный тренд с ковариатами: ρ(t) = t * exp(-β·c)"""
    name = "linear_covariate"
    def func(self, x, t, c=0):
        return t / np.exp(x[2] * c)

    def invFunc(self, x, z0, c=0):  return z0 * np.exp(x[2] * c) / x[1]

    def d_func_beta(self, t, x, c=0):
        return -c * t / np.exp(x[2] * c)

    def d2_func_beta(self, t, x, c=0):
        return (c ** 2) * t / np.exp(x[2] * c)


class PowerWithCovariateTrend(Trend):
    name = "power_covariate" # (t/e^bx)^y
    def func(self, x, t, c=0):
        return np.power(t / np.exp(x[3] * c), x[2])

    def invFunc(self, x, z0, c=0):
        return np.power(z0/x[1], 1 / x[2]) * np.exp(x[3] * c)

    def _base(self, t, x, c=0):
        """Базовая функция f = t^gamma * exp(-beta*c*gamma)"""
        return np.power(t, x[2]) * np.exp(-x[3] * c * x[2])

    def d_func_gamma(self, t, x, c=0):
        if t == 0:
            return 0
        gamma, beta = x[2], x[3]
        f = self._base(t, x, c)
        return f * (np.log(t) - beta * c)

    def d_func_beta(self, t, x, c=0):
        if t == 0:
            return 0
        gamma, beta = x[2], x[3]
        f = self._base(t, x, c)
        return -c * gamma * f

    def d2_func_gamma(self, t, x, c=0):
        if t == 0:
            return 0
        gamma, beta = x[2], x[3]
        f = self._base(t, x, c)
        return f * (np.log(t) - beta * c) ** 2

    def d2_func_beta(self, t, x, c=0):
        if t == 0:
            return 0
        gamma, beta = x[2], x[3]
        f = self._base(t, x, c)
        return (c * gamma) ** 2 * f

    def d2_func_gamma_beta(self, t, x, c=0):
        if t == 0:
            return 0
        gamma, beta = x[2], x[3]
        f = self._base(t, x, c)
        return -c * f * (gamma * (np.log(t) - beta * c) + 1)