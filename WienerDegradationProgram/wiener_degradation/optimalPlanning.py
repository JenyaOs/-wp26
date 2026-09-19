import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize, differential_evolution
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

from wiener_degradation import (
    WienerModel, LinearTrend, PowerTrend,
    PowerWithCovariateTrend, LinearWithCovariateTrend, Wiener_IMF, Wiener_ConditionIMF
)

def unpack_variables(x, n_stresses):
    stresses = x[:n_stresses]
    weights  = x[n_stresses : 2 * n_stresses]
    return stresses, weights

def build_data_per_unit(current_stresses, current_weights, n_units, n_stresses, params, z0, Tmax):
    power_exp = max(1e-6, abs(params[2]))
    units_per_stress = max(1, n_units // n_stresses)

    stresses_per_unit = []
    weights_per_unit = []
    times_per_unit = []

    for i in range(n_units):
        stress_idx = min(i // units_per_stress, n_stresses - 1)
        S = current_stresses[stress_idx]

        t_limit_calc = np.power(z0 , 1.0 / params[2]) / np.exp(-params[3] * S)
        t_limit = min(t_limit_calc, float(Tmax))
        t_limit = max(1.0, t_limit)

        n_points = int(np.floor(t_limit))
        unit_times = list(range(0, n_points, 1))
        if len(unit_times) == 0:
            unit_times = [0.0]

        stresses_per_unit.append(S)
        n_on_this_level = units_per_stress
        if stress_idx == n_stresses - 1:
            n_on_this_level = n_units - stress_idx * units_per_stress
        weights_per_unit.append(current_weights[stress_idx] / n_on_this_level)
        times_per_unit.append(unit_times)

    return np.array(stresses_per_unit), np.array(weights_per_unit), times_per_unit

def d_optimality_criterion(x, n_stresses, params, z0, Tmax, trend_class, n_units):
    current_stresses, current_weights = unpack_variables(x, n_stresses)

    current_weights = np.abs(current_weights)
    current_weights = current_weights / np.sum(current_weights)

    stresses_per_unit, weights_per_unit, times_per_unit = build_data_per_unit(
        current_stresses, current_weights, n_units, n_stresses, params, z0, Tmax
    )

    trend = trend_class()
    model = WienerModel(x=params, trend=trend, z0=z0)

    try:
        fim_obj = Wiener_IMF(model, n_units, stresses_per_unit, weights_per_unit, times_per_unit)
        fim = np.array(fim_obj.getIMF())

        if fim.ndim < 2:
            return 1e10

        fim_reg = fim + 1e-8 * np.eye(fim.shape[0])

        det_fim = np.linalg.det(fim_reg)
        if det_fim <= 0:
            return 1e10
        return -np.log(det_fim)
    except Exception as e:
        return 1e10

def a_optimality_criterion(x, n_stresses, params, z0, Tmax, trend_class, n_units):
    current_stresses, current_weights = unpack_variables(x, n_stresses)

    current_weights = np.abs(current_weights)
    current_weights = current_weights / np.sum(current_weights)

    stresses_per_unit, weights_per_unit, times_per_unit = build_data_per_unit(
        current_stresses, current_weights, n_units, n_stresses, params, z0, Tmax
    )

    trend = trend_class()
    model = WienerModel(x=params, trend=trend, z0=z0)

    try:
        fim_obj = Wiener_IMF(model, n_units, stresses_per_unit, weights_per_unit, times_per_unit)
        fim = np.array(fim_obj.getIMF())

        if fim.ndim < 2:
            return 1e10

        fim_reg = fim + 1e-6 * np.eye(fim.shape[0])

        fim_inv = np.linalg.inv(fim_reg)
        trace_val = np.trace(fim_inv)

        if not np.isfinite(trace_val) or trace_val < 0:
            return 1e10

        return trace_val
    except Exception as e:
        return 1e10

def build_optimal_plans(true_theta, cov_values, weights, z0, Tmax, trend_class, n_units):
    n_stresses = len(weights)
    x0 = np.concatenate([cov_values, weights])

    print(f"[DEBUG] Начальная точка x0: {x0}")
    print(f"[DEBUG] Сумма весов: {np.sum(x0[n_stresses:2*n_stresses])}")

    # Границы для дифференциальной эволюции
    s_lo, s_hi = 0.5, 8.0
    w_lo, w_hi = 0.01, 0.99

    bounds_de = (
        [(s_lo, s_hi)] * n_stresses +
        [(w_lo, w_hi)] * n_stresses
    )

    print(f"[DEBUG] len(x0)={len(x0)}, len(bounds)={len(bounds_de)}")

    # Ограничение для локальной оптимизации
    constraints = {
        'type': 'eq',
        'fun': lambda x: np.sum(x[n_stresses : 2 * n_stresses]) - 1.0
    }

    # === Д-ПЛАН: Глобальная оптимизация ===
    print("\nОптимизация Д-плана (D-optimality)...")
    print("  Этап 1: Глобальный поиск (differential_evolution)...")

    def d_criterion_wrapper(x):
        # Нормализация весов внутри обёртки
        x_copy = x.copy()
        x_copy[n_stresses:2*n_stresses] = np.abs(x_copy[n_stresses:2*n_stresses])
        x_copy[n_stresses:2*n_stresses] /= np.sum(x_copy[n_stresses:2*n_stresses])
        return d_optimality_criterion(x_copy, n_stresses, true_theta, z0, Tmax, trend_class, n_units)

    res_D_global = differential_evolution(
        d_criterion_wrapper,
        bounds_de,
        maxiter=20,
        popsize=15,
        tol=1e-6,
        seed=42,
        disp=True
    )

    print(f"  Глобальный поиск завершён. Лучшее значение: {res_D_global.fun:.6f}")
    print("  Этап 2: Локальная точная настройка (SLSQP)...")

    res_D = minimize(
        d_optimality_criterion, res_D_global.x,
        args=(n_stresses, true_theta, z0, Tmax, trend_class, n_units),
        method='SLSQP',
        bounds=bounds_de,
        constraints=constraints,
        options={'maxiter': 100, 'ftol': 1e-9}
    )

    # === А-ПЛАН: Глобальная оптимизация ===
    """
    print("\nОптимизация А-плана (A-optimality)...")
    print("  Этап 1: Глобальный поиск (differential_evolution)...")

    def a_criterion_wrapper(x):
        x_copy = x.copy()
        x_copy[n_stresses:2*n_stresses] = np.abs(x_copy[n_stresses:2*n_stresses])
        x_copy[n_stresses:2*n_stresses] /= np.sum(x_copy[n_stresses:2*n_stresses])
        return a_optimality_criterion(x_copy, n_stresses, true_theta, z0, Tmax, trend_class, n_units)

    res_A_global = differential_evolution(
        a_criterion_wrapper,
        bounds_de,
        maxiter=20,
        popsize=15,
        tol=1e-6,
        seed=42,
        disp=True
    )

    print(f"  Глобальный поиск завершён. Лучшее значение: {res_A_global.fun:.6f}")
    print("  Этап 2: Локальная точная настройка (SLSQP)...")

    res_A = minimize(
        a_optimality_criterion, res_A_global.x,
        args=(n_stresses, true_theta, z0, Tmax, trend_class, n_units),
        method='SLSQP',
        bounds=bounds_de,
        constraints=constraints,
        options={'maxiter': 100, 'ftol': 1e-9}
    )
    """
    return {'D_plan': res_D}

def format_result(res, n_stresses, z0, Tmax, true_theta, n_units):
    x_opt = res.x
    stresses = x_opt[:n_stresses]
    weights  = x_opt[n_stresses : 2 * n_stresses]
    weights  = weights / np.sum(weights)

    power_exp = max(1e-6, abs(true_theta[2]))
    times_per_level = []
    for S in stresses:
        t_limit_calc = np.power(z0 * np.exp(true_theta[3] * S), 1.0 / power_exp)
        t_limit = min(int(t_limit_calc), Tmax)
        t_limit = max(1, t_limit)
        times_per_level.append(list(range(0, t_limit, 1)))

    return {
        'status': 'Success' if res.success else 'Failed',
        'message': res.message,
        'stresses': np.round(stresses, 4),
        'weights': np.round(weights, 4),
        'measurement_times_per_level': times_per_level,
        'criterion_value': np.round(res.fun, 6) if np.isfinite(res.fun) else res.fun
    }

if __name__ == "__main__":
    true_theta = np.array([ 0.2, 1, 1.3, -0.15])
    z0 = 100
    Tmax = 20
    trend_class = PowerWithCovariateTrend

    cov_values = np.array([0.6, 8.0])
    weights    = np.array([0.5, 0.5])
    n_units    = 100

    print(f"Истинные параметры: {true_theta}")
    print(f"z0 = {z0}, Tmax = {Tmax}")
    print(f"Начальные нагрузки: {cov_values}")
    print(f"Начальные веса:     {weights}")
    print(f"Число изделий:      {n_units}\n")

    optimal_plans = build_optimal_plans(
        true_theta, cov_values, weights, z0, Tmax, trend_class, n_units
    )

    n_stresses = len(cov_values)
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("=" * 60)

    for plan_name, res in optimal_plans.items():
        plan_data = format_result(res, n_stresses, z0, Tmax, true_theta, n_units)
        print(f"\n[{plan_name}]")
        print(f"  Статус: {plan_data['status']} ({plan_data['message']})")
        print(f"  Оптимальные нагрузки: {plan_data['stresses']}")
        print(f"  Оптимальные веса:     {plan_data['weights']}")
        print(f"  Моменты измерений по уровням нагрузки:")
        for idx, times in enumerate(plan_data['measurement_times_per_level']):
            print(f"    Уровень {idx} (S={plan_data['stresses'][idx]:.4f}): "
                  f"[0, 1, ..., {times[-1]}] ({len(times)} точек)")
        print(f"  Значение критерия:    {plan_data['criterion_value']}")