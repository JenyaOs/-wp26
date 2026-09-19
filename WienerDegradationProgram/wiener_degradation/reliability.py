"""Расчёт функции надёжности."""

import numpy as np
from scipy import stats
from .models import WienerModel
from .data import DegradationDataGenerator


class ReliabilityCalculator:
    """Калькулятор функции надёжности R(t) методом Монте-Карло."""

    def __init__(self, model: WienerModel, generator: DegradationDataGenerator,
                 model_type: str, covariate_values: list = None):
        self.model = model
        self.generator = generator
        self.model_type = model_type
        self.covariate_values = covariate_values or [0]


    def compute(self, n_sim, t_grid, covariate_val=None):
        """
        Вычисление функции надёжности R(t).

        Возвращает: (t_grid, R_mean, R_lower, R_upper)
        """
        survived = np.zeros((n_sim, len(t_grid)))

        for i in range(n_sim):
            # Создаём новый экземпляр модели для каждой симуляции
            model_copy = WienerModel(self.model.x, self.model.trend, self.model.z)

            if self.model_type == "Линейная":
                sample = self.generator.generate_linear(1, self.model.x, self.model.z)
            elif self.model_type == "Степенная":
                sample = self.generator.generate_power(1, self.model.x, self.model.z)
            elif self.model_type == "Линейная + ковариаты":
                cov = covariate_val if covariate_val is not None else np.random.choice(self.covariate_values)
                sample = self.generator.generate_custom_covariate(
                    [cov], [1], self.model.x, self.model.z, "linear"
                )
            else:
                cov = covariate_val if covariate_val is not None else np.random.choice(self.covariate_values)
                sample = self.generator.generate_custom_covariate(
                    [cov], [1], self.model.x, self.model.z, "power"
                )

            times = sample[0][0]
            values = sample[2][0]

            failure_time = np.inf
            for t, v in zip(times, values):
                if v >= self.model.z:
                    failure_time = t
                    break

            for j, t in enumerate(t_grid):
                survived[i, j] = 1 if failure_time > t else 0

        R_mean = np.mean(survived, axis=0)
        R_lower, R_upper = self._wilson_confidence_interval(survived, n_sim)

        return t_grid, R_mean, R_lower, R_upper

    def _wilson_confidence_interval(self, survived, n_sim, alpha=0.05):
        """Доверительный интервал Уилсона для биномиальной пропорции."""
        R_lower = np.zeros(survived.shape[1])
        R_upper = np.zeros(survived.shape[1])
        z = stats.norm.ppf(1 - alpha / 2)

        for j in range(survived.shape[1]):
            k = int(np.sum(survived[:, j]))
            p_hat = k / n_sim
            denom = 1 + z**2 / n_sim
            center = (p_hat + z**2 / (2 * n_sim)) / denom
            margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_sim)) / n_sim) / denom
            R_lower[j] = max(0, center - margin)
            R_upper[j] = min(1, center + margin)

        return R_lower, R_upper

    @staticmethod
    def compute_mttf(R_mean, t_vals):
        """Вычисление MTTF (средняя наработка на отказ)."""
        return float(np.sum((R_mean[:-1] + R_mean[1:]) / 2.0 * np.diff(t_vals)))

    @staticmethod
    def find_time_for_R(R_arr, t_arr, target):
        """Найти время, когда R(t) достигает заданного уровня."""
        idx = np.where(R_arr <= target)[0]
        return t_arr[idx[0]] if len(idx) > 0 else np.nan