"""Генерация деградационных данных."""

import numpy as np
from .trends import LinearTrend, PowerTrend, LinearWithCovariateTrend, PowerWithCovariateTrend
from .models import WienerModel


class DegradationDataGenerator:
    """Генератор данных деградационных процессов."""

    def __init__(self, model: WienerModel):
        self.model = model


    def get_delta(self, time, x, c=0):
        """Генерация приращений деградации."""
        array_d = np.zeros(len(time) - 1)
        for i in range(len(time) - 1):
            delta_ro = self.model.f_ro(time[i + 1], c) - self.model.f_ro(time[i], c)
            array_d[i] = self.model.increment(delta_ro)
        return array_d

    def get_values_unlimited(self, delta, time, z0):
        """Получение значений деградации без ограничения порогом."""
        value = []
        times = []
        for i in range(len(delta)):
            value.append([0])
            times.append(time[i])
            for j in range(len(delta[i])):
                value[i].append(value[i][j] + delta[i][j])
        return value, delta, times

    def get_values_limited(self, delta, time, z0):
        """Получение значений деградации с ограничением порогом."""
        value = []
        times = []
        for i in range(len(delta)):
            value.append([0])
            times.append(time[i])
            for j in range(len(delta[i])):
                if value[i][j] + delta[i][j] <= z0:
                    value[i].append(value[i][j] + delta[i][j])
                else:
                    times[i] = time[i][0:j + 1]
                    delta[i] = delta[i][0:j + 1]
                    break
        return value, delta, times

    def generate_linear(self, M, x, z0, covariate, time):
        """Генерация данных для линейной модели."""
        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_unlimited(delta, time, z0)

        return [time, delta, value, covariate]

    def generate_power(self, M, x, z0, covariate, time):
        """Генерация данных для степенной модели."""

        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_unlimited(delta, time, z0)
        return [time, delta, value, covariate]

    def generate_linear_with_covariate(self, M, x, z0, covariate, time):
        """Генерация данных для линейной модели с ковариатами."""
        #time = [np.linspace(0, maxTime, momentsCount) for _ in range(M)]

        #for i in range(M):
        #    time.append(time)

        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_unlimited(delta, time, z0)
        return [time, delta, value, covariate]

    def generate_power_with_covariate(self, M, x, z0,  covariate, time):
        """Генерация данных для степенной модели с ковариатами."""
        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_unlimited(delta, time, z0)
        return [time, delta, value, covariate]

    def generate_custom_covariate(self, covariate_values, covariate_counts,  x, z0, trend_type, maxTime, momentsCount):
        """Генерация данных с пользовательскими ковариатами."""
        times, covariates = [], []
        time = np.linspace(0, maxTime, momentsCount)

        for cov_val, count in zip(covariate_values, covariate_counts):
            for _ in range(count):
                if trend_type == "linear":
                    t_len = 25 * np.exp(min(covariate_values) * x[2]) + 2
                else:
                    t_len = np.power(25 * np.exp(min(covariate_values) * x[3] * x[2]), 1/x[2]) + 2
                times.append(time)
                covariates.append(cov_val)

        delta = [self.get_delta(times[i], x, covariates[i]) for i in range(len(times))]
        value, delta, times = self.get_values_unlimited(delta, times, z0)
        return [times, delta, value, covariates]

    def generate_linear_limited(self, M, x, z0):
        """Генерация данных для линейной модели с ограничением по порогу."""
        time = [np.linspace(0, 40, 100) for _ in range(M)]
        covariate = [0] * M
        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_limited(delta, time, z0)
        return [time, delta, value, covariate]

    def generate_power_limited(self, M, x, z0):
        """Генерация данных для степенной модели с ограничением по порогу."""
        time = [np.linspace(0, 40, 100) for _ in range(M)]
        covariate = [0] * M
        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_limited(delta, time, z0)
        return [time, delta, value, covariate]

    def generate_linear_with_covariate_limited(self, M, x, z0):
        """Генерация данных для линейной модели с ковариатами и ограничением."""
        time = []
        covariate = [(i % 2) * 1.5 + 0.5 for i in range(M)]
        for i in range(M):
            time.append(np.linspace(0, 25 * np.exp(min(covariate) * x[2]) + 2, 20))
        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_limited(delta, time, z0)
        return [time, delta, value, covariate]

    def generate_power_with_covariate_limited(self, M, x, z0):
        """Генерация данных для степенной модели с ковариатами и ограничением."""
        time = []
        covariate = [(i % 2) * 1.5 + 0.5 for i in range(M)]
        for i in range(M):
            t_len = np.power(25 * np.exp(min(covariate) * x[3] * x[2]), 1/x[2]) + 2
            time.append(np.linspace(0, t_len, 20))
        delta = [self.get_delta(time[i], x, covariate[i]) for i in range(M)]
        value, delta, time = self.get_values_limited(delta, time, z0)
        return [time, delta, value, covariate]