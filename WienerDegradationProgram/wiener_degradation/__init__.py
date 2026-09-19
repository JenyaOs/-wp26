"""
Wiener Degradation Model Library
================================
Библиотека для моделирования деградационных процессов на основе винеровского процесса.

Основные возможности:
- 4 типа моделей (линейная, степенная, с ковариатами)
- Генерация деградационных данных
- Оценка параметров методом максимального правдоподобия (MLE)
- Расчёт функции надёжности R(t) с доверительными интервалами
- Информационная матрица Фишера (IMF)
- Оптимальное планирование экспериментов (D-оптимальность)
"""

from .trends import (
    LinearTrend,
    PowerTrend,
    LinearWithCovariateTrend,
    PowerWithCovariateTrend,
)
from .models import WienerModel
from .data import DegradationDataGenerator
from .estimation import ParameterEstimator
from .reliability import ReliabilityCalculator
from .imf import Wiener_IMF, Wiener_ConditionIMF
#from .planning import OptimalPlanningProcedure

__version__ = "1.0.0"
__all__ = [
    "LinearTrend", "PowerTrend", "LinearWithCovariateTrend", "PowerWithCovariateTrend",
    "WienerModel", "DegradationDataGenerator", "ParameterEstimator",
    "ReliabilityCalculator", "Wiener_IMF", "Wiener_ConditionIMF", "OptimalPlanningProcedure", ""
]