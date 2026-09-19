"""Основная модель Винера для деградации."""

import random
import numpy as np
from scipy.optimize import minimize
import mpmath


class WienerModel:
    """
    Винеровская деградационная модель.

    Параметры:
        x: список параметров [σ, μ, ...]
        trend: объект тренда (LinearTrend, PowerTrend и т.д.)
        z0: порог отказа
    """

    def __init__(self, x, trend, z0):
        self.x = list(x)
        self.trend = trend
        self.z = z0
        self.data = None

    @property
    def sigma(self):
        return self.x[0]

    @property
    def mu(self):
        return self.x[1]

    def set_params(self, x):
        self.x = list(x)

    def f_ro(self, t, c=0):
        """Значение тренда ρ(t) в момент t."""
        return self.trend.func(self.x, t, c)

    def inv_f_ro(self, z0, c=0):
        """Значение времени (t) в момент достижения z0."""
        return self.trend.invFunc(self.x, z0, c)

    def increment(self, delta_ro):
        """Приращение винеровского процесса."""
        theta1 = self.x[1] * delta_ro
        theta2 = self.x[0] * np.sqrt(delta_ro)
        return random.normalvariate(theta1, theta2)

    def log_likelihood(self, x):
        """Логарифмическая функция правдоподобия."""
        self.set_params(x)
        time = self.data[0]
        delta = self.data[1]
        cov = self.data[3]
        n = len(time)

        if x[0] <= 0 or x[1] <= 0:
            return 1e+10

        f = 0
        for i in range(n):
            f += (len(time[i]) - 1) * (0.5 * np.log(2 * np.pi) + np.log(x[0]))
            s1, s2 = 0, 0
            for j in range(len(time[i]) - 1):
                delta_ro = self.f_ro(time[i][j + 1], cov[i]) - self.f_ro(time[i][j], cov[i])
                theta1 = x[1] * delta_ro
                theta2 = x[0] * np.sqrt(2 * delta_ro)
                if theta1 <= 0 or theta2 <= 0:
                    return 1e+10
                s1 += np.power((delta[i][j] - theta1) / theta2, 2)
                s2 += np.log(delta_ro) / 2
            f += s1 + s2
        return f

    def estimate_parameters(self, x0, data):
        """Оценка параметров методом MLE."""
        self.data = data
        res = minimize(self.log_likelihood, x0, method='L-BFGS-B')
        return res.x


    def conditional_log_likelihood(self, x):
        """Условная логарифмическая функция правдоподобия с ТОЧНОЙ формулой."""
        self.set_params(x)
        z = self.z
        time = self.data[0]
        delta = self.data[1]
        covariates = self.data[3]
        n = len(time)

        if x[0] <= 0 or x[1] <= 0:
            return 1e+10

        from scipy.stats import norm

        neg_log_lik = 0.0

        for i in range(n):
            ro = self.f_ro(time[i][-1], covariates[i])
            z_last = sum(delta[i])

            # 1. Отрицательный логарифм плотности (безусловная часть)
            neg_log_pdf = (
                0.5 * np.log(2 * np.pi)
                + np.log(x[0])
                + 0.5 * np.log(ro)
                + np.power(z_last - x[1] * ro, 2) / (2 * np.power(x[0], 2) * ro)
            )

            # 2. ТОЧНАЯ функция надёжности (Inverse Gaussian Survival Function)
            # R(t) = Φ(C) - exp(2μz/σ²) · Φ(C2)
            C = (z - x[1] * ro) / (x[0] * np.sqrt(ro))
            C2 = (-z - x[1] * ro) / (x[0] * np.sqrt(ro))

            exp_arg = 2 * x[1] * z / (x[0]**2)
            if exp_arg > 700:  # Защита от переполнения
                exp_term = np.exp(700)
            else:
                exp_term = np.exp(exp_arg)

            term1 = norm.cdf(C)
            term2 = exp_term * norm.cdf(C2)

            R = term1 - term2
            R = max(R, 1e-300)  # Защита от log(0)

            log_R = np.log(R)

            # 3. Условное правдоподобие: -log L_cond = -log PDF + log R
            neg_log_lik += neg_log_pdf - log_R

        return neg_log_lik

    def estimate_conditional_parameters(self, x0, data):
        """Оценка условных параметров."""
        self.data = data
        res = minimize(self.conditional_log_likelihood, x0, method='L-BFGS-B')
        return res.x