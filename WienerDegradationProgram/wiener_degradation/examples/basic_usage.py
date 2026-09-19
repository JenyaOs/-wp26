"""Пример использования библиотеки wiener_degradation с визуализацией."""

import numpy as np
import matplotlib.pyplot as plt
from itertools import takewhile
from wiener_degradation import (
    WienerModel, LinearTrend, LinearWithCovariateTrend, PowerWithCovariateTrend, PowerTrend,
    DegradationDataGenerator, ParameterEstimator, ReliabilityCalculator, Wiener_IMF, Wiener_ConditionIMF
)

# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def get_params_to_IMF(model, count_obj, cov, weight, z0, maxTime, momentsCount):
    times_list, cov_list = [], []
    count_per_group = [int(count_obj * weight[i]) for i in range(len(weight) - 1)]
    count_per_group.append(count_obj - sum(count_per_group))

    times = np.linspace(0, maxTime, momentsCount)
    print(times)
    for i in range(len(cov)):
        print(cov[i],model.inv_f_ro(z0, cov[i]))
        t_max = min(maxTime, model.inv_f_ro(z0, cov[i]))
        times_cur = list(takewhile(lambda x: x <= t_max, times))
        for j in range(count_per_group[i]):
            times_list.append(times_cur)
            cov_list.append(cov[i])

    return times_list, cov_list


def plot_results(model, generator, sample, x_true, x_est, model_name, z0, max_time):
    """Строит графики траекторий деградации и функции надёжности с учётом ковариат."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    times_list = sample[0]
    deltas_list = sample[1]
    cov_list = sample[3] if len(sample) > 3 else None

    # Определяем уникальные значения ковариат
    unique_covs = []
    if cov_list is not None and len(cov_list) > 0:
        # Преобразуем ковариаты в строки для уникальности (на случай, если это списки)
        cov_strings = [str(c) for c in cov_list]
        unique_cov_strings = list(dict.fromkeys(cov_strings))  # Сохраняем порядок
        unique_covs = [cov_list[cov_strings.index(cs)] for cs in unique_cov_strings]

    num_to_plot = min(100, len(times_list))

    # Цветовая палитра для разных нагрузок
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_covs), 1)))

    # --- График 1: Траектории деградации ---
    plotted_covs = set()

    for i in range(num_to_plot):
        t_i = list(times_list[i])
        z_i = np.cumsum([0] + list(deltas_list[i]))

        # Определяем цвет в зависимости от ковариаты
        if unique_covs and cov_list is not None:
            cov_val = cov_list[i]
            cov_idx = unique_covs.index(cov_val) if cov_val in unique_covs else 0
            color = colors[cov_idx]

            # Добавляем в легенду только первый раз для каждой ковариаты
            cov_label = f"c={cov_val}" if not isinstance(cov_val, list) else f"c={cov_val}"
            if cov_label not in plotted_covs:
                ax1.plot(t_i, z_i, marker='.', alpha=0.6, linewidth=1, color=color, label=cov_label)
                plotted_covs.add(cov_label)
            else:
                ax1.plot(t_i, z_i, marker='.', alpha=0.6, linewidth=1, color=color)
        else:
            ax1.plot(t_i, z_i, marker='.', alpha=0.6, linewidth=1, color='steelblue')

    ax1.axhline(z0, color='red', linestyle='--', linewidth=2, label=f'Порог отказа (z0={z0})')
    ax1.set_title(f"{model_name}: Траектории деградации (первые {num_to_plot})")
    ax1.set_xlabel("Время")
    ax1.set_ylabel("Накопленная деградация")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- График 2: Функция надёжности ---
    try:
        rel_calc = ReliabilityCalculator(model, generator, model_name)
        t_grid = np.linspace(0.1, max_time * 1.5, 200)

        # 1. Рисуем оценку (на основе найденных параметров)
        model.set_params(x_est)
        t_vals_est, R_mean, R_lower, R_upper = rel_calc.compute(n_sim=500, t_grid=t_grid)
        ax2.plot(t_vals_est, R_mean, label='Оценённая R(t)', color='blue', linewidth=2)
        ax2.fill_between(t_vals_est, R_lower, R_upper, color='blue', alpha=0.2, label='95% ДИ')

        # 2. Рисуем истинную функцию для сравнения
        model.set_params(x_true)
        t_vals_true, R_true, _, _ = rel_calc.compute(n_sim=500, t_grid=t_grid)
        ax2.plot(t_vals_true, R_true, label='Истинная R(t)', color='green', linestyle=':', linewidth=2)

        ax2.set_title(f"{model_name}: Функция надёжности")
        ax2.set_xlabel("Время")
        ax2.set_ylabel("R(t)")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    except Exception as e:
        ax2.text(0.5, 0.5, f"Ошибка расчёта надёжности:\n{e}", ha='center', va='center', color='red')
        ax2.set_title(f"{model_name}: Функция надёжности (Ошибка)")

    # Возвращаем истинные параметры модели
    model.set_params(x_true)

    plt.tight_layout()
    plt.show()

# ==============================================================================
# ПРИМЕРЫ
# ==============================================================================
def example_linear_model():
    """Пример для линейной модели."""
    print("=" * 60)
    print("Линейная модель деградации")
    print("=" * 60)

    sigma, mu, z0 = 0.2, 1.0, 50
    x_true = [sigma, mu]
    maxTime, momentsCount, M_count = 21, 21, 100

    trend = LinearTrend()
    model = WienerModel(x=x_true, trend=trend, z0=z0)

    times_list, cov_list = get_params_to_IMF(model, M_count, [0], [1], z0, maxTime, momentsCount)

    generator = DegradationDataGenerator(model)
    sample = generator.generate_linear(M=M_count, x=x_true, z0=z0, covariate=cov_list, time=times_list)

    estimator = ParameterEstimator(model)
    x0 = [0.1, 0.5]
    estimated = estimator.estimate(x0, sample)

    print(f"Истинные параметры:   σ={sigma:.3f}, μ={mu:.3f}")
    print(f"Оценённые параметры:  σ={estimated[0]:.3f}, μ={estimated[1]:.3f}")
    print(f"Относительные ошибки: {estimator.compute_errors(x_true, estimated)}\n")

    # Расчет ИМФ
    model.x = x_true
    imf_calculator = Wiener_IMF(model, M_count, [0], [1], times_list)
    analytical_fim = np.array(imf_calculator.getIMF())
    print("Аналитическая ИМФ (Wiener_IMF):\n", analytical_fim.astype('int'))

    # Расчет УИМФ
    condImf_calculator = Wiener_ConditionIMF(model, M_count, [0], [1], times_list, z0)
    cond_analytical_fim = np.array(condImf_calculator.getIMF())
    print("Условная аналитическая ИМФ (Wiener_ConditionIMF):\n", cond_analytical_fim.astype('int'))

    # === ВИЗУАЛИЗАЦИЯ ===
    plot_results(model, generator, sample, x_true, estimated, "Линейная модель", z0, maxTime)


def example_power_model():
    """Пример для степенной модели."""
    print("\n" + "=" * 60)
    print("Степенная модель деградации")
    print("=" * 60)

    sigma, mu, gamma, z0 = 0.2, 1.0, 1.5, 50
    x_true = [sigma, mu, gamma]
    maxTime, momentsCount, M_count = 21, 21, 100 # Увеличил M_count для лучшей оценки

    trend = PowerTrend()
    model = WienerModel(x=x_true, trend=trend, z0=z0)

    times_list, cov_list = get_params_to_IMF(model, M_count, [0], [1], z0, maxTime, momentsCount)

    generator = DegradationDataGenerator(model)
    sample = generator.generate_power(M=M_count, x=x_true, z0=z0, covariate=cov_list, time=times_list)

    estimator = ParameterEstimator(model)
    x0 = [0.1, 0.5, 1.0]
    estimated = estimator.estimate(x0, sample)

    print(f"Истинные параметры:   σ={sigma:.3f}, μ={mu:.3f}, γ={gamma:.3f}")
    print(f"Оценённые параметры:  σ={estimated[0]:.3f}, μ={estimated[1]:.3f}, γ={estimated[2]:.3f}")
    print(f"Относительные ошибки: {estimator.compute_errors(x_true, estimated)}\n")

    # Расчет ИМФ
    model.x = x_true
    imf_calculator = Wiener_IMF(model, M_count, [0], [1], times_list) # Исправил вес на [1] для корректности
    analytical_fim = np.array(imf_calculator.getIMF())
    print("Аналитическая ИМФ (Wiener_IMF):\n", analytical_fim.astype('int'))
    print("Определитель ИМФ:", np.linalg.det(analytical_fim))

    # Расчет УИМФ
    condImf_calculator = Wiener_ConditionIMF(model, M_count, [0], [1], times_list, z0)
    cond_analytical_fim = np.array(condImf_calculator.getIMF())
    print("Условная аналитическая ИМФ (Wiener_ConditionIMF):\n", cond_analytical_fim.astype('int'))
    print("Определитель УИМФ:", np.linalg.det(cond_analytical_fim))

    # === ВИЗУАЛИЗАЦИЯ ===
    plot_results(model, generator, sample, x_true, estimated, "Степенная модель", z0, maxTime)


def example_linear_model_covariate():
    """Пример для линейной модели с ковариатами."""
    print("\n" + "=" * 60)
    print("Линейная модель деградации с ковариатами")
    print("=" * 60)

    sigma, mu, beta, z0 = 0.2, 1.0, 1.2, 50
    x_true = [sigma, mu, beta]
    maxTime, momentsCount, M_count = 100, 21, 100 # Увеличил M_count для лучшей оценки

    trend = LinearWithCovariateTrend()
    model = WienerModel(x=x_true, trend=trend, z0=z0)

    cov = [0.5, 2.0]
    weight = [0.5, 0.5]
    times_list, cov_list = get_params_to_IMF(model, M_count, cov, weight, z0, maxTime, momentsCount)

    generator = DegradationDataGenerator(model)
    sample = generator.generate_linear_with_covariate(M=M_count, x=x_true, z0=z0, covariate=cov_list, time=times_list)

    estimator = ParameterEstimator(model)
    x0 = [0.1, 0.5, 0.8] # Исправлена опечатка: было 0,8
    estimated = estimator.estimate(x0, sample)

    print(f"Истинные параметры:   σ={sigma:.3f}, μ={mu:.3f}, β={beta:.3f}")
    print(f"Оценённые параметры:  σ={estimated[0]:.3f}, μ={estimated[1]:.3f}, β={estimated[2]:.3f}")
    print(f"Относительные ошибки: {estimator.compute_errors(x_true, estimated)}\n")

    # Расчет ИМФ
    model.x = x_true
    imf_calculator = Wiener_IMF(model, M_count, cov, weight, times_list)
    analytical_fim = np.array(imf_calculator.getIMF())
    print("Аналитическая ИМФ (Wiener_IMF):\n", analytical_fim.astype('int'))
    print("Определитель ИМФ:", np.linalg.det(analytical_fim))

    # Расчет УИМФ
    condImf_calculator = Wiener_ConditionIMF(model, M_count, cov, weight, times_list, z0)
    cond_analytical_fim = np.array(condImf_calculator.getIMF())
    print("Условная аналитическая ИМФ (Wiener_ConditionIMF):\n", cond_analytical_fim.astype('int'))
    print("Определитель УИМФ:", np.linalg.det(cond_analytical_fim))

    # === ВИЗУАЛИЗАЦИЯ ===
    plot_results(model, generator, sample, x_true, estimated, "Линейная с ковариатами", z0, maxTime)


def example_power_model_covariate():
    """Пример для степенной модели с ковариатами."""
    print("\n" + "=" * 60)
    print("Степенная модель деградации с ковариатами")
    print("=" * 60)

    sigma, mu, gamma, beta, z0 = 0.2, 1, 1.3, -0.15, 100
    x_true = [sigma, mu, gamma, beta]
    maxTime, momentsCount, M_count = 20, 41, 40 # Увеличил M_count для лучшей оценки

    trend = PowerWithCovariateTrend()
    model = WienerModel(x=x_true, trend=trend, z0=z0)

    cov = [8. ,   0.628]
    weight = [0.5, 0.5]
    times_list, cov_list = get_params_to_IMF(model, M_count, cov, weight, z0, maxTime, momentsCount)

    generator = DegradationDataGenerator(model)
    # Примечание: убедитесь, что в DegradationDataGenerator есть метод generate_power_with_covariate
    # или что generate_power корректно обрабатывает ковариаты. Если нет, используйте соответствующий метод.
    sample = generator.generate_power_with_covariate(M=M_count, x=x_true, z0=z0, covariate=cov_list, time=times_list)

    estimator = ParameterEstimator(model)
    x0 = [0.1, 1.2, 1.0, -0.1]
    estimated = estimator.estimate(x0, sample)

    print(f"Истинные параметры:   σ={sigma:.3f}, μ={mu:.3f}, γ={gamma:.3f}, β={beta:.3f}")
    print(f"Оценённые параметры:  σ={estimated[0]:.3f}, μ={estimated[1]:.3f}, γ={estimated[2]:.3f}, β={estimated[3]:.3f}")
    print(f"Относительные ошибки: {estimator.compute_errors(x_true, estimated)}\n")

    # Расчет ИМФ
    model.x = x_true
    imf_calculator = Wiener_IMF(model, M_count, cov, weight, times_list)
    analytical_fim = np.array(imf_calculator.getIMF())
    print("Аналитическая ИМФ (Wiener_IMF):\n", analytical_fim.astype('int'))
    print("Определитель ИМФ:", np.linalg.det(analytical_fim))

    # Расчет УИМФ
    condImf_calculator = Wiener_ConditionIMF(model, M_count, cov, weight, times_list, z0)
    cond_analytical_fim = np.array(condImf_calculator.getIMF())
    print("Условная аналитическая ИМФ (Wiener_ConditionIMF):\n", cond_analytical_fim.astype('int'))
    print("Определитель УИМФ:", np.linalg.det(cond_analytical_fim))
    #print(sample[3])
    # === ВИЗУАЛИЗАЦИЯ ===
    plot_results(model, generator, sample, x_true, estimated, "Степенная с ковариатами", z0, maxTime)

import numpy as np

import numpy as np

def monteCarlo(N_max, filename="monte_carlo_results2_1000.txt"):
    sigma, mu, gamma, beta, z0 = 0.2, 1, 1.3, -0.15, 100
    x_true = [sigma, mu, gamma, beta]
    maxTime, momentsCount, M_count = 20, 41, 40

    trend = PowerWithCovariateTrend()
    model = WienerModel(x=x_true, trend=trend, z0=z0)

    cov = [2., 0.5]
    weight = [0.5, 0.5]
    times_list, cov_list = get_params_to_IMF(model, M_count, cov, weight, z0, maxTime, momentsCount)

    estimated_array = []

    with open(filename, "w", encoding="utf-8") as f:
        f.write("Итерация; sigma; mu; gamma; beta\n")

        for i in range(N_max):
            generator = DegradationDataGenerator(model)
            sample = generator.generate_power_with_covariate(M=M_count, x=x_true, z0=z0, covariate=cov_list, time=times_list)
            estimator = ParameterEstimator(model)
            x0 = [0.5, 0.5, 1.8, 0.8]
            estimated = estimator.estimate(x0, sample)
            estimated_array.append(estimated)

            print(f"[{i+1}/{N_max}] Оценённые параметры: σ={estimated[0]:.3f}, μ={estimated[1]:.3f}, γ={estimated[2]:.3f}, β={estimated[3]:.3f}")

            # ИСПРАВЛЕННАЯ СТРОКА (убрана лишняя скобка {{)
            f.write(f"{estimated[0]:.3f}; {estimated[1]:.3f}; {estimated[2]:.3f}; {estimated[3]:.3f}\n")

        estimated_np = np.array(estimated_array)
        means = np.mean(estimated_np, axis=0)
        stds = np.std(estimated_np, axis=0)

        f.write("\n--- Итоговая статистика (Монте-Карло) ---\n")
        f.write(f"План х={cov[0]:.3f}, {cov[1]:.3f}\n")
        f.write(f"Истинные значения: σ={x_true[0]:.3f}, μ={x_true[1]:.3f}, γ={x_true[2]:.3f}, β={x_true[3]:.3f}\n")
        f.write(f"Средние значения: σ={means[0]:.3f}, μ={means[1]:.3f}, γ={means[2]:.3f}, β={means[3]:.3f}\n")
        f.write(f"СКО (std):        σ={stds[0]:.3f}, μ={stds[1]:.3f}, γ={stds[2]:.3f}, β={stds[3]:.3f}\n")

    return estimated_array

def monteCarlo2(N_max, filename="monte_carlo_results3_1000.txt"):
    sigma, mu, gamma, beta, z0 = 0.2, 1, 1.3, -0.15, 100
    x_true = [sigma, mu, gamma, beta]
    maxTime, momentsCount, M_count = 20, 41, 40

    trend = PowerWithCovariateTrend()
    model = WienerModel(x=x_true, trend=trend, z0=z0)

    cov = [8., 0.6]
    weight = [0.5, 0.5]
    times_list, cov_list = get_params_to_IMF(model, M_count, cov, weight, z0, maxTime, momentsCount)

    estimated_array = []

    with open(filename, "w", encoding="utf-8") as f:
        f.write("Итерация; sigma; mu; gamma; beta\n")

        for i in range(N_max):
            generator = DegradationDataGenerator(model)
            sample = generator.generate_power_with_covariate(M=M_count, x=x_true, z0=z0, covariate=cov_list, time=times_list)
            estimator = ParameterEstimator(model)
            x0 = [0.5, 0.5, 1.8, 0.8]
            estimated = estimator.estimate(x0, sample)
            estimated_array.append(estimated)

            print(f"[{i+1}/{N_max}] Оценённые параметры: σ={estimated[0]:.3f}, μ={estimated[1]:.3f}, γ={estimated[2]:.3f}, β={estimated[3]:.3f}")

            # ИСПРАВЛЕННАЯ СТРОКА (убрана лишняя скобка {{)
            f.write(f"{estimated[0]:.3f}; {estimated[1]:.3f}; {estimated[2]:.3f}; {estimated[3]:.3f}\n")

        estimated_np = np.array(estimated_array)
        means = np.mean(estimated_np, axis=0)
        stds = np.std(estimated_np, axis=0)

        f.write("\n--- Итоговая статистика (Монте-Карло) ---\n")
        f.write(f"План х={cov[0]:.3f}, {cov[1]:.3f}\n")
        f.write(f"Истинные значения: σ={x_true[0]:.3f}, μ={x_true[1]:.3f}, γ={x_true[2]:.3f}, β={x_true[3]:.3f}\n")
        f.write(f"Средние значения: σ={means[0]:.3f}, μ={means[1]:.3f}, γ={means[2]:.3f}, β={means[3]:.3f}\n")
        f.write(f"СКО (std):        σ={stds[0]:.3f}, μ={stds[1]:.3f}, γ={stds[2]:.3f}, β={stds[3]:.3f}\n")

    return estimated_array


if __name__ == "__main__":
    # Настройка стиля графиков для красоты
    plt.style.use('seaborn-v0_8-whitegrid')

    #example_linear_model()
    #example_power_model()
    #example_linear_model_covariate()
    #example_power_model_covariate()
    print(monteCarlo(1000))
    print(monteCarlo2(1000))