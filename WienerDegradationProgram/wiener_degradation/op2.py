import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

from wiener_degradation import (
    WienerModel, PowerWithCovariateTrend, Wiener_IMF
)

# ==========================================
# ПАРАМЕТРИЗАЦИЯ МОМЕНТОВ ВРЕМЕНИ С МИН. ШАГОМ
# ==========================================
def params_to_times_with_min_step(u, t_max, dt_min):
    n_points = len(u) + 1
    u_clipped = np.clip(u, -50, 50)
    raw_deltas = np.log1p(np.exp(u_clipped))

    min_total = (n_points - 1) * dt_min
    available_range = t_max - min_total

    if available_range <= 1e-8:
        return np.arange(n_points) * dt_min

    deltas = dt_min + raw_deltas / np.sum(raw_deltas) * available_range
    times = np.concatenate([[0.0], np.cumsum(deltas)])
    times = np.minimum(times, t_max)
    return times

def times_to_params_with_min_step(times, t_max, dt_min):
    n_points = len(times)
    if n_points < 2:
        return np.zeros(1)

    deltas = np.diff(times)
    free_deltas = deltas - dt_min
    free_deltas = np.maximum(free_deltas, 1e-8)
    u = np.log(free_deltas / np.sum(free_deltas) * (n_points - 1))
    return u

# ==========================================
# ПОСТРОЕНИЕ ДАННЫХ
# ==========================================
def build_data_per_unit(current_stresses, current_weights, n_units, n_stresses,
                        params, z0, Tmax, common_times, dt_min):
    power_exp = max(1e-6, abs(params[2]))
    units_per_stress = max(1, n_units // n_stresses)

    stresses_per_unit = []
    weights_per_unit = []
    times_per_unit = []
    completion_status = []

    t_limits = []
    for S in current_stresses:
        t_limit_calc = np.power(z0, 1.0 / power_exp) / np.exp(-params[3] * S)
        t_limit = min(t_limit_calc, float(Tmax))
        t_limit = max(dt_min, t_limit)
        t_limits.append(t_limit)

    for i in range(n_units):
        stress_idx = min(i // units_per_stress, n_stresses - 1)
        S = current_stresses[stress_idx]
        t_limit = t_limits[stress_idx]

        t_limit_calc = np.power(z0 * np.exp(params[3] * S), 1.0 / power_exp)
        is_completed = 1 if t_limit_calc <= Tmax else 0
        completion_status.append(is_completed)

        level_times = np.array(common_times)
        level_times = level_times[level_times <= t_limit + 1e-9]

        if len(level_times) == 0:
            level_times = np.array([0.0])
        elif level_times[0] != 0.0:
            level_times = np.concatenate([[0.0], level_times])

        stresses_per_unit.append(S)
        n_on_this_level = units_per_stress
        if stress_idx == n_stresses - 1:
            n_on_this_level = n_units - stress_idx * units_per_stress
        weights_per_unit.append(current_weights[stress_idx] / n_on_this_level)
        times_per_unit.append(level_times.tolist())

    return (np.array(stresses_per_unit), np.array(weights_per_unit),
            times_per_unit, np.array(completion_status), t_limits)

def compute_fim_and_metrics(current_stresses, current_weights, n_units, n_stresses,
                            params, z0, Tmax, trend_class, common_times, dt_min):
    stresses_per_unit, weights_per_unit, times_per_unit, completion_status, t_limits = \
        build_data_per_unit(current_stresses, current_weights, n_units, n_stresses,
                            params, z0, Tmax, common_times, dt_min)

    trend = trend_class()
    model = WienerModel(x=params, trend=trend, z0=z0)

    fim_obj = Wiener_IMF(model, n_units, stresses_per_unit, weights_per_unit, times_per_unit)
    fim = np.array(fim_obj.getIMF())

    fim_reg = fim + 1e-6 * np.eye(fim.shape[0])

    completion_rate = np.mean(completion_status)
    avg_time_points = np.mean([len(t) for t in times_per_unit])

    return fim_reg, completion_rate, avg_time_points, t_limits

def compute_pure_criterion(fim_reg, criterion_type='D'):
    """Вычисляет ЧИСТЫЙ критерий (без штрафов) для контроля сходимости."""
    try:
        if criterion_type == 'D':
            det_fim = np.linalg.det(fim_reg)
            if det_fim <= 0:
                return np.inf
            return -np.log(det_fim)
        else:  # 'A'
            fim_inv = np.linalg.inv(fim_reg)
            return np.trace(fim_inv)
    except:
        return np.inf

# ==========================================
# КРИТЕРИИ СО ШТРАФАМИ
# ==========================================
def d_criterion_with_penalty(current_stresses, current_weights, n_units, n_stresses,
                             params, z0, Tmax, trend_class, common_times, dt_min):
    fim_reg, completion_rate, avg_time_points, _ = compute_fim_and_metrics(
        current_stresses, current_weights, n_units, n_stresses, params, z0, Tmax,
        trend_class, common_times, dt_min
    )
    try:
        det_fim = np.linalg.det(fim_reg)
        if det_fim <= 0:
            return 1e10
        d_crit = -np.log(det_fim)
        penalty = 10.0 * (1.0 - completion_rate)
        time_penalty = 5.0 / max(1.0, avg_time_points)
        return d_crit + penalty + time_penalty
    except:
        return 1e10

def a_criterion_with_penalty(current_stresses, current_weights, n_units, n_stresses,
                             params, z0, Tmax, trend_class, common_times, dt_min):
    fim_reg, completion_rate, avg_time_points, _ = compute_fim_and_metrics(
        current_stresses, current_weights, n_units, n_stresses, params, z0, Tmax,
        trend_class, common_times, dt_min
    )
    try:
        fim_inv = np.linalg.inv(fim_reg)
        a_crit = np.trace(fim_inv)
        if not np.isfinite(a_crit) or a_crit < 0:
            return 1e10
        penalty = 100.0 * (1.0 - completion_rate)
        time_penalty = 50.0 / max(1.0, avg_time_points)
        return a_crit + penalty + time_penalty
    except:
        return 1e10

# ==========================================
# ШАГИ КООРДИНАТНОГО СПУСКА
# ==========================================
def optimize_stresses_only(current_weights, n_stresses, true_theta, z0, Tmax,
                           trend_class, n_units, criterion_func, common_times, dt_min):
    def objective(x_stresses):
        return criterion_func(x_stresses, current_weights, n_units, n_stresses,
                              true_theta, z0, Tmax, trend_class, common_times, dt_min)

    bounds = [(0.5, 8.0)] * n_stresses
    x0 = np.array([6.0, 2.5])[:n_stresses]

    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                   options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
    return res.x

def optimize_weights_only(current_stresses, n_stresses, true_theta, z0, Tmax,
                          trend_class, n_units, criterion_func, common_times, dt_min):
    def objective(x_weights):
        x_weights = np.abs(x_weights)
        x_weights = x_weights / np.sum(x_weights)
        return criterion_func(current_stresses, x_weights, n_units, n_stresses,
                              true_theta, z0, Tmax, trend_class, common_times, dt_min)

    bounds = [(0.01, 0.99)] * n_stresses
    x0 = np.array([0.5, 0.5])[:n_stresses]

    res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                   options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})

    opt_weights = np.abs(res.x)
    opt_weights = opt_weights / np.sum(opt_weights)
    return opt_weights

def optimize_times_only(current_stresses, current_weights, n_stresses, true_theta,
                        z0, Tmax, trend_class, n_units, criterion_func,
                        common_times, dt_min, n_measurements=10):
    t_max_common = float(Tmax)

    if len(common_times) == n_measurements:
        u0 = times_to_params_with_min_step(common_times, t_max_common, dt_min)
    else:
        init_times = np.linspace(0, t_max_common, n_measurements)
        init_times = np.maximum(init_times, np.arange(n_measurements) * dt_min)
        u0 = times_to_params_with_min_step(init_times, t_max_common, dt_min)

    def objective(u):
        new_common_times = params_to_times_with_min_step(u, t_max_common, dt_min)
        return criterion_func(current_stresses, current_weights, n_units, n_stresses,
                              true_theta, z0, Tmax, trend_class, new_common_times, dt_min)

    res = minimize(objective, u0, method='Nelder-Mead',
                   options={'maxiter': 1500, 'xatol': 1e-3, 'fatol': 1e-4})

    opt_common_times = params_to_times_with_min_step(res.x, t_max_common, dt_min)
    return opt_common_times

# ==========================================
# ГЛАВНАЯ ФУНКЦИЯ С АВТОМАТИЧЕСКОЙ ОСТАНОВКОЙ
# ==========================================
def build_optimal_plans_until_convergence(true_theta, cov_values, weights, z0, Tmax,
                                          trend_class, n_units, dt_min=0.5,
                                          max_iterations=50, tol=1e-4,
                                          n_measurements=10, criterion_type='D'):
    """
    Оптимизирует план до сходимости.

    Остановка происходит, когда:
      - улучшение чистого критерия меньше tol
      - или достигнут лимит max_iterations
      - или критерий начал ухудшаться (откат к лучшему плану)
    """
    n_stresses = len(weights)
    power_exp = max(1e-6, abs(true_theta[2]))
    criterion_func = d_criterion_with_penalty if criterion_type == 'D' else a_criterion_with_penalty

    print("=" * 80)
    print(f"ОПТИМИЗАЦИЯ {criterion_type}-ПЛАНА (до сходимости)")
    print(f"Макс. итераций: {max_iterations}, порог сходимости: {tol}")
    print("=" * 80)

    current_stresses = cov_values.copy()
    current_weights = weights.copy()

    # Начальный общий массив моментов времени
    common_times_init = np.linspace(0, Tmax, n_measurements)
    current_common_times = common_times_init.copy()

    # Сохраняем начальный план как лучший
    best_stresses = current_stresses.copy()
    best_weights = current_weights.copy()
    best_common_times = current_common_times.copy()

    # Вычисляем начальный чистый критерий
    fim_reg_init, completion_rate_init, _, t_limits_init = compute_fim_and_metrics(
        current_stresses, current_weights, n_units, n_stresses, true_theta, z0, Tmax,
        trend_class, current_common_times, dt_min
    )
    best_pure_criterion = compute_pure_criterion(fim_reg_init, criterion_type)
    prev_pure_criterion = best_pure_criterion

    print(f"\nНАЧАЛЬНЫЙ ПЛАН:")
    print(f"  Нагрузки:       {current_stresses}")
    print(f"  Веса:           {current_weights}")
    print(f"  {criterion_type}-критерий (чистый): {best_pure_criterion:.6f}")
    print(f"  Завершённость:  {completion_rate_init:.2%}")
    print(f"  t_limit: {t_limits_init}")
    print(f"  Общий массив моментов: [{', '.join(f'{t:.3f}' for t in current_common_times)}]")

    # === ЦИКЛ ДО СХОДИМОСТИ ===
    iteration = 0
    no_improvement_count = 0
    stop_reason = None

    print(f"\n{'─' * 80}")
    print(f"{'Итер':>5} | {'Чистый критерий':>18} | {'Улучшение':>12} | {'Статус':<20}")
    print(f"{'─' * 80}")

    while iteration < max_iterations:
        iteration += 1

        # Сохраняем состояние перед итерацией (для отката)
        prev_stresses = current_stresses.copy()
        prev_weights = current_weights.copy()
        prev_common_times = current_common_times.copy()

        # Шаг 1: Оптимизация нагрузок
        current_stresses = optimize_stresses_only(
            current_weights, n_stresses, true_theta, z0, Tmax, trend_class,
            n_units, criterion_func, current_common_times, dt_min
        )

        # Шаг 2: Оптимизация весов
        current_weights = optimize_weights_only(
            current_stresses, n_stresses, true_theta, z0, Tmax, trend_class,
            n_units, criterion_func, current_common_times, dt_min
        )

        # Шаг 3: Оптимизация моментов времени
        current_common_times = optimize_times_only(
            current_stresses, current_weights, n_stresses, true_theta, z0, Tmax,
            trend_class, n_units, criterion_func, current_common_times,
            dt_min, n_measurements
        )

        # Вычисляем чистый критерий
        fim_reg, completion_rate, _, t_limits = compute_fim_and_metrics(
            current_stresses, current_weights, n_units, n_stresses, true_theta, z0, Tmax,
            trend_class, current_common_times, dt_min
        )
        current_pure_criterion = compute_pure_criterion(fim_reg, criterion_type)

        improvement = prev_pure_criterion - current_pure_criterion

        # Проверяем улучшение
        if current_pure_criterion < best_pure_criterion - 1e-10:
            # Нашли лучший план — сохраняем
            best_pure_criterion = current_pure_criterion
            best_stresses = current_stresses.copy()
            best_weights = current_weights.copy()
            best_common_times = current_common_times.copy()
            best_completion_rate = completion_rate
            best_t_limits = t_limits.copy()
            status = f"✓ улучшено (best={best_pure_criterion:.4f})"
            no_improvement_count = 0
        elif improvement < tol:
            # Нет существенного улучшения
            no_improvement_count += 1
            status = f"⏸ нет улучшения ({no_improvement_count})"

            # Если несколько итераций подряд нет улучшения — останавливаемся
            if no_improvement_count >= 2:
                stop_reason = f"сходимость (улучшение {improvement:.2e} < {tol})"
                print(f"{iteration:>5} | {current_pure_criterion:>18.6f} | {improvement:>12.2e} | {status}")
                break
        else:
            # Критерий ухудшился — откат
            status = f"↶ откат (ухудшение на {-improvement:.4f})"
            current_stresses = prev_stresses
            current_weights = prev_weights
            current_common_times = prev_common_times
            no_improvement_count += 1

            if no_improvement_count >= 2:
                stop_reason = "откат к лучшему плану"
                print(f"{iteration:>5} | {current_pure_criterion:>18.6f} | {improvement:>12.2e} | {status}")
                break

        print(f"{iteration:>5} | {current_pure_criterion:>18.6f} | {improvement:>12.2e} | {status}")

        prev_pure_criterion = current_pure_criterion

    if stop_reason is None:
        if iteration >= max_iterations:
            stop_reason = f"достигнут лимит {max_iterations} итераций"
        else:
            stop_reason = "сходимость"

    print(f"{'─' * 80}")
    print(f"Остановка: {stop_reason}")
    print(f"Всего итераций: {iteration}")
    print(f"Лучший {criterion_type}-критерий: {best_pure_criterion:.6f}")
    print(f"Улучшение от начального: {prev_pure_criterion - best_pure_criterion:.6f} "
          f"({(prev_pure_criterion - best_pure_criterion)/abs(prev_pure_criterion)*100:.2f}%)")

    # Возвращаем ЛУЧШИЙ найденный план
    return {
        'stresses': best_stresses,
        'weights': best_weights,
        'common_times': best_common_times,
        't_limits': best_t_limits,
        'criterion': best_pure_criterion,
        'criterion_init': prev_pure_criterion,
        'completion_rate': best_completion_rate,
        'completion_rate_init': completion_rate_init,
        'stresses_init': cov_values.copy(),
        'weights_init': weights.copy(),
        'common_times_init': common_times_init.copy(),
        't_limits_init': t_limits_init.copy(),
        'n_iterations': iteration,
        'stop_reason': stop_reason,
        'criterion_type': criterion_type
    }

# ==========================================
# ФОРМАТИРОВАНИЕ И ВЫВОД
# ==========================================
def print_comparison_table(plan_data, n_stresses, dt_min):
    criterion_type = plan_data['criterion_type']
    plan_name = f"{criterion_type}-план"

    print(f"\n{'─' * 100}")
    print(f"[{plan_name}] — итераций: {plan_data['n_iterations']}, причина остановки: {plan_data['stop_reason']}")
    print(f"{'─' * 100}")

    print(f"\n{'Параметр':<35} {'Начальный план':<25} {'Итоговый план':<25} {'Δ':<15}")
    print(f"{'─' * 100}")

    for i in range(n_stresses):
        s_init = plan_data['stresses_init'][i]
        s_final = plan_data['stresses'][i]
        delta = s_final - s_init
        print(f"Нагрузка {i:<25} {s_init:<25.4f} {s_final:<25.4f} {delta:+.4f}")

    for i in range(n_stresses):
        w_init = plan_data['weights_init'][i]
        w_final = plan_data['weights'][i]
        delta = w_final - w_init
        print(f"Вес {i:<29} {w_init:<25.4f} {w_final:<25.4f} {delta:+.4f}")

    for i in range(n_stresses):
        tl_init = plan_data['t_limits_init'][i]
        tl_final = plan_data['t_limits'][i]
        delta = tl_final - tl_init
        print(f"t_limit (уровень {i}){'':<16} {tl_init:<25.4f} {tl_final:<25.4f} {delta:+.4f}")

    c_init = plan_data['criterion_init']
    c_final = plan_data['criterion']
    delta_c = c_final - c_init
    pct = (c_init - c_final) / abs(c_init) * 100 if c_init != 0 else 0
    print(f"{'─' * 100}")
    print(f"{criterion_type}-критерий{'':<22} {c_init:<25.6f} {c_final:<25.6f} {delta_c:+.6f} ({pct:+.2f}%)")

    comp_init = plan_data['completion_rate_init']
    comp_final = plan_data['completion_rate']
    print(f"Доля завершённых процессов{'':<10} {comp_init:<25.2%} {comp_final:<25.2%} {comp_final-comp_init:+.2%}")

    print(f"\n{'─' * 100}")
    print("ОБЩИЙ МАССИВ МОМЕНТОВ ВРЕМЕНИ:")
    print(f"{'─' * 100}")

    t_init = plan_data['common_times_init']
    t_final = plan_data['common_times']

    print(f"\n  НАЧАЛЬНЫЙ ({len(t_init)} точек):")
    print(f"    [{', '.join(f'{t:.3f}' for t in t_init)}]")
    if len(t_init) >= 2:
        deltas_init = np.diff(t_init)
        print(f"    Шаги: мин={deltas_init.min():.3f}, ср={deltas_init.mean():.3f}, макс={deltas_init.max():.3f}")

    print(f"\n  ИТОГОВЫЙ ({len(t_final)} точек):")
    print(f"    [{', '.join(f'{t:.3f}' for t in t_final)}]")
    if len(t_final) >= 2:
        deltas_final = np.diff(t_final)
        print(f"    Шаги: мин={deltas_final.min():.3f}, ср={deltas_final.mean():.3f}, макс={deltas_final.max():.3f}")
        print(f"    Минимальный шаг (ограничение): {dt_min:.3f}")

    print(f"\n{'─' * 100}")
    print("ФАКТИЧЕСКИЕ МОМЕНТЫ ИЗМЕРЕНИЙ ПО УРОВНЯМ НАГРУЗКИ:")
    print(f"{'─' * 100}")

    for idx in range(n_stresses):
        S = plan_data['stresses'][idx]
        t_limit = plan_data['t_limits'][idx]

        used_times = t_final[t_final <= t_limit + 1e-9]
        used_times_init = t_init[t_init <= plan_data['t_limits_init'][idx] + 1e-9]

        print(f"\n  Уровень {idx} (S={S:.4f}, t_limit={t_limit:.3f}):")
        print(f"    Использовано моментов: {len(used_times)} из {len(t_final)}")
        print(f"    НАЧАЛЬНЫЕ: [{', '.join(f'{t:.3f}' for t in used_times_init)}]")
        print(f"    ИТОГОВЫЕ:  [{', '.join(f'{t:.3f}' for t in used_times)}]")
        if len(used_times) > 0:
            print(f"    Последний момент (достижение z0): {used_times[-1]:.3f}")

    print(f"\n{'─' * 100}")

# ==========================================
# ТОЧКА ВХОДА
# ==========================================
if __name__ == "__main__":
    true_theta = np.array([0.5, 0.8, 1.5, 0.5])
    z0 = 30
    Tmax = 20
    trend_class = PowerWithCovariateTrend

    cov_values = np.array([6.0, 2.5])
    weights = np.array([0.5, 0.5])
    n_units = 20

    dt_min = 0.5
    n_measurements = 21
    max_iterations = 50   # максимум итераций
    tol = 1e-4            # порог сходимости

    print(f"Истинные параметры: {true_theta}")
    print(f"z0 = {z0}, Tmax = {Tmax}")
    print(f"Начальные нагрузки: {cov_values}")
    print(f"Начальные веса:     {weights}")
    print(f"Число изделий:      {n_units}")
    print(f"Минимальный шаг:    {dt_min}")
    print(f"Число моментов:     {n_measurements}")
    print(f"Макс. итераций:     {max_iterations}")
    print(f"Порог сходимости:   {tol}\n")

    # Д-план
    d_plan = build_optimal_plans_until_convergence(
        true_theta, cov_values, weights, z0, Tmax, trend_class, n_units,
        dt_min=dt_min, max_iterations=max_iterations, tol=tol,
        n_measurements=n_measurements, criterion_type='D'
    )

    '''
    a_plan = build_optimal_plans_until_convergence(
        true_theta, cov_values, weights, z0, Tmax, trend_class, n_units,
        dt_min=dt_min, max_iterations=max_iterations, tol=tol,
        n_measurements=n_measurements, criterion_type='A'
    )'''

    n_stresses = len(cov_values)
    print("\n" + "=" * 100)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 100)

    for plan_data in [d_plan]:
        print_comparison_table(plan_data, n_stresses, dt_min)