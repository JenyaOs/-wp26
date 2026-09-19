"""Оценка параметров модели."""

import numpy as np
from .models import WienerModel


class ParameterEstimator:
    """Класс для оценки параметров модели методом MLE."""

    def __init__(self, model: WienerModel):
        self.model = model

    def estimate(self, x0, data):
        """Оценка параметров."""
        return self.model.estimate_parameters(x0, data)

    def estimate_conditional(self, x0, data):
        """Оценка условных параметров."""
        return self.model.estimate_conditional_parameters(x0, data)

    def get_initial_guess(self, model_type):
        """Получить начальное приближение для оптимизации."""
        if model_type == "Линейная":
            return [0.1, 0.5]
        elif model_type == "Степенная":
            return [0.1, 0.5, 1.0]
        elif model_type == "Линейная + ковариаты":
            return [0.1, 0.5, 0.3]
        elif model_type == "Степенная + ковариаты":
            return [0.1, 0.5, 1.0, 0.3]
        else:
            raise ValueError(f"Неизвестный тип модели: {model_type}")

    def compute_errors(self, true_params, estimated_params):
        """Вычисление относительных ошибок оценки."""
        return [abs(e - t) / t * 100 for e, t in zip(estimated_params, true_params)]