# Wiener Degradation Model

Библиотека для моделирования деградационных процессов на основе винеровского процесса.

## Возможности

- 📐 **4 типа моделей**: линейная, степенная, с ковариатами
- 🎲 **Генерация данных** деградационных траекторий
- 🔍 **Оценка параметров** методом максимального правдоподобия (MLE)
- 📉 **Функция надёжности** R(t) с доверительными интервалами
- 📊 **Информационная матрица Фишера** (IMF)
- 🎯 **Оптимальное планирование** экспериментов (D-оптимальность)
- 🌐 **Веб-интерфейс** на Streamlit

## Установка

```bash
pip install -e .
```

## Быстрый старт

```python
from wiener_degradation import (
    WienerModel, LinearTrend, DegradationDataGenerator,
    ParameterEstimator, ReliabilityCalculator
)

# Создаём модель
trend = LinearTrend()
model = WienerModel(x=[0.2, 1.0], trend=trend, z0=50)

# Генерируем данные
generator = DegradationDataGenerator(model)
sample = generator.generate_linear(M=100, x=[0.2, 1.0], z0=50)

# Оцениваем параметры
estimator = ParameterEstimator(model)
estimated = estimator.estimate(x0=[0.1, 0.5], data=sample)
print(f"Оценённые параметры: σ={estimated[0]:.3f}, μ={estimated[1]:.3f}")

# Считаем функцию надёжности
rel_calc = ReliabilityCalculator(model, generator, "Линейная")
t_grid, R_mean, R_lower, R_upper = rel_calc.compute(n_sim=1000, t_grid=np.linspace(0, 100, 200))
```

## Веб-интерфейс

```bash
streamlit run app.py
```

## Лицензия

MIT