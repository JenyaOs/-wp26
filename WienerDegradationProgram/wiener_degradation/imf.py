"""Информационная матрица Фишера (IMF) для винеровской модели."""

import numpy as np
import mpmath
from wiener_degradation import WienerModel
import time as tm

class Wiener_IMF:
    """Расчёт информационной матрицы Фишера."""

    def __init__(self, model: WienerModel, n, cov, weight, time):
        self.model = model
        self.weight = weight
        self.time = time
        self.covariates = self._convert_to_array(cov, weight, n)

    def _convert_to_array(self, cov, weight, n):
        array = []
        for i in range(len(cov) - 1):
            for _ in range(int(weight[i] * n)):
                array.append(cov[i])
        # Дополняем до n, если из-за округления не хватило
        while len(array) < n:
            array.append(cov[-1])
        return array[:n]

    def getIMF(self):
        trend_name = self.model.trend.name

        if trend_name == 'linear':
            return [[self.I11(), self.I12()], [self.I12(), self.I22()]]
        elif trend_name == 'linear_covariate':
            return [[self.I11(), self.I12(), self.I14()],
                    [self.I12(), self.I22(), self.I24()],
                    [self.I14(), self.I24(), self.I44()]]
        elif trend_name == 'power':
            return [[self.I11(), self.I12(), self.I13()],
                    [self.I12(), self.I22(), self.I23()],
                    [self.I13(), self.I23(), self.I33()]]
        elif trend_name == 'power_covariate':
            return [[self.I11(), self.I12(), self.I13(), self.I14()],
                    [self.I12(), self.I22(), self.I23(), self.I24()],
                    [self.I13(), self.I23(), self.I33(), self.I34()],
                    [self.I14(), self.I24(), self.I34(), self.I44()]]
        else:
            raise ValueError(f"Неизвестный тип тренда: {trend_name}")

    def I11(self):
        time = self.time
        n = len(time)
        x = self.model.x
        return 2 * sum([len(time[i]) - 1 for i in range(n)]) / np.power(x[0], 2)

    def I12(self):
        return 0.0

    def I13(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro = self.model.trend.d_func_gamma(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_gamma(time[i][j], x, cov[i])
                if delta_ro != 0:
                    s += der_ro / delta_ro
        return s / x[0]

    def I14(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_beta(time[i][j], x, cov[i])
                if delta_ro != 0:
                    s += der_ro / delta_ro
        return s / x[0]

    def I22(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                s += delta_ro
        return s / np.power(x[0], 2)

    def I23(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                der_ro = self.model.trend.d_func_gamma(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_gamma(time[i][j], x, cov[i])
                s += der_ro
        return s * x[1] / np.power(x[0], 2)

    def I24(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                der_ro = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_beta(time[i][j], x, cov[i])
                s += der_ro
        return s * x[1] / np.power( x[0], 2)

    def I33(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro = self.model.trend.d_func_gamma(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_gamma(time[i][j], x, cov[i])
                if delta_ro != 0:
                    s += np.power(der_ro, 2) * (0.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s

    def I34(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro_g = self.model.trend.d_func_gamma(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_gamma(time[i][j], x, cov[i])
                der_ro_b = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_beta(time[i][j], x, cov[i])
                if delta_ro != 0:
                    s += der_ro_g * der_ro_b * (0.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s

    def I44(self):
        time = self.time
        n = len(time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            for j in range(len(time[i]) - 1):
                delta_ro = self.model.f_ro(time[i][j + 1], cov[i]) - self.model.f_ro(time[i][j], cov[i])
                der_ro = self.model.trend.d_func_beta(time[i][j + 1], x, cov[i]) - self.model.trend.d_func_beta(time[i][j], x, cov[i])
                if delta_ro != 0:
                    s += np.power(der_ro, 2) * (0.5 / np.power(delta_ro, 2) + np.power(x[1] / x[0], 2) / delta_ro)
        return s

class Wiener_ConditionIMF1:
    """Расчёт условной информационной матрицы Фишера."""

    def __init__(self, model: WienerModel, n, cov, weight, time, z0):
        self.model = model
        self.weight = weight
        self.time = time
        self.z0 = z0
        self.covariates = self._convert_to_array(cov, weight, n)

    def _convert_to_array(self, cov, weight, n):
        array = []
        print(cov)
        for i in range(len(cov) - 1):
            for j in range(int(weight[i] * n)):
                array.append(cov[i])
        while len(array) < n:
            array.append(cov[-1])
        return array[:n]

    def getIMF(self):
        IMF1 = np.array(self.getIMF_part1())
        IMF2 = np.array(self.getIMF_part2())
        return IMF1 + IMF2

    def getIMF_part1(self):
        trend_name = self.model.trend.name
        if trend_name == 'linear':
            return [[self.I11(), self.I12()], [self.I12(), self.I22()]]
        elif trend_name == 'linear_covariate':
            return [[self.I11(), self.I12(), self.I14()], [self.I12(), self.I22(), self.I24()], [self.I14(), self.I24(), self.I44()]]
        elif trend_name == 'power':
            return [[self.I11(), self.I12(), self.I13()], [self.I12(), self.I22(), self.I23()], [self.I13(), self.I23(), self.I33()]]
        elif trend_name == 'power_covariate':
            return [[self.I11(), self.I12(), self.I13(), self.I14()],
                    [self.I12(), self.I22(), self.I23(), self.I24()],
                    [self.I13(), self.I23(), self.I33(), self.I34()],
                    [self.I14(), self.I24(), self.I34(), self.I44()]]

    def getIMF_part2(self):
        trend_name = self.model.trend.name
        if trend_name == 'linear':
            return [[self.I11_F(), self.I12_F()], [self.I12_F(), self.I22_F()]]
        elif trend_name == 'linear_covariate':
            return [[self.I11_F(), self.I12_F(), self.I14_F()], [self.I12_F(), self.I22_F(), self.I24_F()], [self.I14_F(), self.I24_F(), self.I44_F()]]
        elif trend_name == 'power':
            return [[self.I11_F(), self.I12_F(), self.I13_F()], [self.I12_F(), self.I22_F(), self.I23_F()], [self.I13_F(), self.I23_F(), self.I33_F()]]
        elif trend_name == 'power_covariate':
            return [[self.I11_F(), self.I12_F(), self.I13_F(), self.I14_F()],
                    [self.I12_F(), self.I22_F(), self.I23_F(), self.I24_F()],
                    [self.I13_F(), self.I23_F(), self.I33_F(), self.I34_F()],
                    [self.I14_F(), self.I24_F(), self.I34_F(), self.I44_F()]]

    def I11(self):
        n = len(self.time)
        x = self.model.x
        return n/ np.power(x[0], 2)

    def I12(self):
        return 0.0

    def I13(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            delta_ro = self.model.f_ro(self.time[i][-1], cov[i])
            der_ro = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            if delta_ro != 0:
                s += der_ro / delta_ro
        return s / x[0]

    def I14(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            delta_ro = self.model.f_ro(self.time[i][-1], cov[i])
            der_ro = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            if delta_ro != 0:
                s += der_ro / delta_ro
        return s / x[0]

    def I22(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            delta_ro = self.model.f_ro(self.time[i][-1], cov[i])
            s += delta_ro
        return s / np.power(x[0], 2)

    def I23(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            der_ro = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            s += der_ro
        return s * x[1] / np.power( x[0], 2)

    def I24(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0
        for i in range(n):
            der_ro = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            s += der_ro
        return s * x[1] / np.power(x[0], 2)

    def I33(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            d_rho_g = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            if rho > 1e-12:
                term1 = (d_rho_g ** 2) / (x[0] ** 2 * rho)          # 1/(σ² ρ)
                term2 = (x[1] ** 2) * (d_rho_g ** 2) / (2 * x[0] ** 2 * rho ** 2)  # μ²/(2 σ² ρ²)
                s += term1 + term2
        return s

    def I44(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            d_rho_b = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            if rho > 1e-12:
                term1 = (d_rho_b ** 2) / (x[0] ** 2 * rho)
                term2 = (x[1] ** 2) * (d_rho_b ** 2) / (2 * x[0] ** 2 * rho ** 2)
                s += term1 + term2
        return s

    def I34(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            d_rho_g = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            d_rho_b = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            if rho > 1e-12:
                term1 = (d_rho_g * d_rho_b) / (x[0] ** 2 * rho)
                term2 = (x[1] ** 2) * (d_rho_g * d_rho_b) / (2 * x[0] ** 2 * rho ** 2)
                s += term1 + term2
        return s

    def F(self, i):
        time = self.time[i]
        x = self.model.x
        cov = self.covariates
        z0 = self.z0
        ro_val = self.model.f_ro(time[-1], cov)

        arg = np.power(z0 - x[1] * ro_val, 2) / (2 * np.power(x[0], 2) * ro_val)
        s = 0.5 + (0.5 / np.sqrt(np.pi)) * mpmath.gammainc(0.5, arg)
        return float(s)

    def C(self, i):
        time = self.time
        x = self.model.x
        cov = self.covariates
        z0 = self.z0
        ro_val = self.model.f_ro(time[i][-1], cov[i])
        return (z0 - x[1] * ro_val) / (x[0] * np.sqrt(2 * ro_val))

    def d_C(self, i, k):
        time = self.time
        x = self.model.x
        cov = self.covariates
        z0 = self.z0
        ro_val = self.model.f_ro(time[i][-1], cov[i])

        if k == 0:
            return -(z0 - x[1] * ro_val) / (np.power(x[0], 2) * np.sqrt(2 * ro_val))
        if k == 1:
            return -np.sqrt(ro_val) / (np.sqrt(2) * x[0])
        if k == 2:
            der_g = self.model.trend.d_func_gamma(time[i][-1], x, cov[i])
            return -der_g * (x[1] + z0 / ro_val) / (x[0] * 2 * np.sqrt(2 * ro_val))
        if k == 3:
            der_b = self.model.trend.d_func_beta(time[i][-1], x, cov[i])
            return -der_b * (x[1] + z0 / ro_val) / (x[0] * 2 * np.sqrt(2 * ro_val))
        return 0.0

    def d2_C(self, i, k, m):
        time = self.time[i]
        x = self.model.x
        c = self.covariates[i]
        z0 = self.z0
        ro_val = self.model.f_ro(time[-1], c)

        if (k == 0 and m == 0):
            return 2 * (z0 - x[1] * ro_val) / (np.power(x[0], 3) * np.sqrt(2 * ro_val))
        if (k == 0 and m == 1) or (k == 1 and m == 0):
            return np.sqrt(ro_val / 2) / np.power(x[0], 2)
        if (k == 0 and m == 2) or (k == 2 and m == 0):
            der_g = self.model.trend.d_func_gamma(time[-1], x, c)
            return der_g * (x[1] + z0 / ro_val) / (2 * np.sqrt(2 * ro_val) * np.power(x[0], 2))
        if (k == 0 and m == 3) or (k == 3 and m == 0):
            der_b = self.model.trend.d_func_beta(time[-1], x, c)
            return der_b * (x[1] + z0 / ro_val) / (2 * np.sqrt(2 * ro_val) * np.power(x[0], 2))
        if (k == 1 and m == 1):
            return 0.0
        if (k == 1 and m == 2) or (k == 2 and m == 1):
            der_g = self.model.trend.d_func_gamma(time[-1], x, c)
            return -0.5 * der_g / (np.sqrt(2 * ro_val) * x[0])
        if (k == 1 and m == 3) or (k == 3 and m == 1):
            der_b = self.model.trend.d_func_beta(time[-1], x, c)
            return -0.5 * der_b / (np.sqrt(2 * ro_val) * x[0])
        if (k == 2 and m == 2):
            der_g = self.model.trend.d_func_gamma(time[-1], x, c)
            d2_g = self.model.trend.d2_func_gamma(time[-1], x, c)
            s1 = 0.5 * np.power(der_g, 2) / np.power(ro_val, 1.5) - d2_g / np.sqrt(ro_val)
            s2 = z0 / ro_val + x[1]
            return s1 * s2 / (2 * np.sqrt(2) * x[0])
        if (k == 2 and m == 3) or (k == 3 and m == 2):
            der_g = self.model.trend.d_func_gamma(time[-1], x, c)
            der_b = self.model.trend.d_func_beta(time[-1], x, c)
            d2_gb = self.model.trend.d2_func_gamma_beta(time[-1], x, c)
            s1 = 0.5 * der_g * der_b / np.power(ro_val, 1.5) - d2_gb / np.sqrt(ro_val)
            s2 = z0 / ro_val + x[1]
            return s1 * s2 / (2 * np.sqrt(2) * x[0])
        if (k == 3 and m == 3):
            der_b = self.model.trend.d_func_beta(time[-1], x, c)
            d2_b = self.model.trend.d2_func_beta(time[-1], x, c)
            s1 = 0.5 * np.power(der_b, 2) / np.power(ro_val, 1.5) - d2_b / np.sqrt(ro_val)
            s2 = z0 / ro_val + x[1]
            return s1 * s2 / (2 * np.sqrt(2) * x[0])
        return 0.0

    def d_F(self, i, k):
        return -self.d_C(i, k) * np.exp(-np.power(self.C(i), 2)) / np.sqrt(np.pi)

    def d2_F(self, i, k, m):
        return (self.C(i) * self.d_C(i, k) * self.d_C(i, m) - self.d2_C(i, k, m)) * np.exp(-np.power(self.C(i), 2)) / np.sqrt(2 * np.pi)

    def d2_lnF(self, k, m):
        total = 0.0
        for i in range(len(self.time)):
            f_val = self.F(i)
            if f_val > 1e-10:  # Защита от деления на ноль
                total += self.d2_F(i, k, m) / f_val - (self.d_F(i, k) * self.d_F(i, m)) / np.power(f_val, 2)
        return float(total)

    def I11_F(self): return self.d2_lnF(0, 0)
    def I12_F(self): return self.d2_lnF(0, 1)
    def I13_F(self): return self.d2_lnF(0, 2)
    def I14_F(self): return self.d2_lnF(0, 3)
    def I22_F(self): return self.d2_lnF(1, 1)
    def I23_F(self): return self.d2_lnF(1, 2)
    def I24_F(self): return self.d2_lnF(1, 3)
    def I33_F(self): return self.d2_lnF(2, 2)
    def I34_F(self): return self.d2_lnF(2, 3)
    def I44_F(self): return self.d2_lnF(3, 3)

import numpy as np
from scipy.stats import norm

import numpy as np
from scipy.stats import norm
from wiener_degradation import WienerModel

import numpy as np
import mpmath
from scipy.stats import norm

import numpy as np
from scipy.stats import norm
from scipy.special import gamma as gamma_func

import numpy as np
import math
import mpmath
import numpy as np
import math


class Wiener_ConditionIMF:
    """Расчёт условной информационной матрицы Фишера."""

    def __init__(self, model, n, cov, weight, time, z0):
        self.model = model
        self.weight = weight
        self.time = time
        self.z0 = z0
        self.covariates = self._convert_to_array(cov, weight, n)

    def _convert_to_array(self, cov, weight, n):
        array = []
        num_groups = min(len(cov) - 1, len(weight))
        for i in range(num_groups):
            count = int(weight[i] * n)
            for _ in range(count):
                array.append(cov[i])
        last_cov = cov[-1] if cov else 0.0
        while len(array) < n:
            array.append(last_cov)
        return array[:n]

    def getIMF(self):
        IMF1 = np.array(self.getIMF_part1())
        IMF2 = np.array(self.getIMF_part2())
        return IMF1 + IMF2

    def getIMF_part1(self):
        trend_name = self.model.trend.name
        if trend_name == 'linear':
            return [[self.I11(), self.I12()], [self.I12(), self.I22()]]
        elif trend_name == 'linear_covariate':
            return [[self.I11(), self.I12(), self.I14()], [self.I12(), self.I22(), self.I24()], [self.I14(), self.I24(), self.I44()]]
        elif trend_name == 'power':
            return [[self.I11(), self.I12(), self.I13()], [self.I12(), self.I22(), self.I23()], [self.I13(), self.I23(), self.I33()]]
        elif trend_name == 'power_covariate':
            return [[self.I11(), self.I12(), self.I13(), self.I14()],
                    [self.I12(), self.I22(), self.I23(), self.I24()],
                    [self.I13(), self.I23(), self.I33(), self.I34()],
                    [self.I14(), self.I24(), self.I34(), self.I44()]]

    def getIMF_part2(self):
        trend_name = self.model.trend.name
        if trend_name == 'linear':
            return [[self.I11_F(), self.I12_F()], [self.I12_F(), self.I22_F()]]
        elif trend_name == 'linear_covariate':
            return [[self.I11_F(), self.I12_F(), self.I14_F()], [self.I12_F(), self.I22_F(), self.I24_F()], [self.I14_F(), self.I24_F(), self.I44_F()]]
        elif trend_name == 'power':
            return [[self.I11_F(), self.I12_F(), self.I13_F()], [self.I12_F(), self.I22_F(), self.I23_F()], [self.I13_F(), self.I23_F(), self.I33_F()]]
        elif trend_name == 'power_covariate':
            return [[self.I11_F(), self.I12_F(), self.I13_F(), self.I14_F()],
                    [self.I12_F(), self.I22_F(), self.I23_F(), self.I24_F()],
                    [self.I13_F(), self.I23_F(), self.I33_F(), self.I34_F()],
                    [self.I14_F(), self.I24_F(), self.I34_F(), self.I44_F()]]

    # === Часть 1: от плотности f(z) ========================================

    def I11(self):
        n = len(self.time)
        x = self.model.x
        return 2 * n / (x[0] ** 2)

    def I12(self):
        return 0.0

    def I13(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            if rho <= 1e-12:
                continue
            der_ro = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            s += der_ro / rho
        return s / x[0]

    def I14(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            if rho <= 1e-12:
                continue
            der_ro = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            s += der_ro / (rho * x[0])
        return s

    def I22(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            s += rho
        return s / (x[0] ** 2)

    def I23(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            der_ro = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            s += der_ro
        return s * x[1] / (x[0] ** 2)

    def I24(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            der_ro = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            s += der_ro
        return s * x[1] / (x[0] ** 2)

    def I33(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            if rho <= 1e-12:
                continue
            d_rho = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            term1 = (d_rho ** 2) *(x[1] ** 2) / (x[0] ** 2 * rho)
            term2 = (d_rho ** 2) / (2 * (x[0] ** 2) * (rho**2) )
            s += term1 + term2
        return s

    def I34(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            if rho <= 1e-12:
                continue
            d_rho_g = self.model.trend.d_func_gamma(self.time[i][-1], x, cov[i])
            d_rho_b = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            term1 = (d_rho_g * d_rho_b) * (x[1] ** 2) / (x[0] ** 2 * rho)
            term2 = (d_rho_g * d_rho_b) /  (2 * (x[0] ** 2) * (rho **2 ) )
            s += term1 + term2
        return s

    def I44(self):
        n = len(self.time)
        x = self.model.x
        cov = self.covariates
        s = 0.0
        for i in range(n):
            rho = self.model.f_ro(self.time[i][-1], cov[i])
            if rho <= 1e-12:
                continue
            d_rho = self.model.trend.d_func_beta(self.time[i][-1], x, cov[i])
            term1 = (d_rho ** 2) *(x[1] ** 2) / (x[0] ** 2 * rho)
            term2 = (d_rho ** 2) / (2 * (x[0] ** 2) * (rho**2) )
            s += term1 + term2
        return s

    # === Часть 2: от функции распределения F(z0) ==========================

    def F(self, i):
        time = self.time[i]
        x = self.model.x
        cov = self.covariates[i]
        z0 = self.z0
        rho = self.model.f_ro(time[-1], cov)
        sigma = x[0]
        mu = x[1]

        if rho <= 1e-12:
            return 1.0 if z0 >= mu * rho else 0.0

        C = (z0 - mu * rho) / (sigma * np.sqrt(2 * rho))
        return 0.5 * (1 + math.erf(C))

    def C(self, i):
        time = self.time[i]
        x = self.model.x
        cov = self.covariates[i]
        z0 = self.z0
        rho = self.model.f_ro(time[-1], cov)
        sigma = x[0]
        mu = x[1]
        if rho <= 1e-12:
            return np.inf if z0 < mu * rho else -np.inf
        return (z0 - mu * rho) / (sigma * np.sqrt(2 * rho))

    def d_C(self, i, k):
        time = self.time[i]
        x = self.model.x
        cov = self.covariates[i]
        z0 = self.z0
        rho = self.model.f_ro(time[-1], cov)
        sigma = x[0]
        mu = x[1]

        if rho <= 1e-12:
            return 0.0

        sqrt_2_rho = np.sqrt(2 * rho)
        if k == 0:  # d/dσ
            return -(z0 - mu * rho) / (sigma ** 2 * sqrt_2_rho)
        elif k == 1:  # d/dμ
            return -np.sqrt(rho / 2) / sigma
        elif k == 2:  # d/dγ
            d_rho = self.model.trend.d_func_gamma(time[-1], x, cov)
            return -d_rho * (mu + z0 / rho) / (2 * sigma * sqrt_2_rho)
        elif k == 3:  # d/dβ
            d_rho = self.model.trend.d_func_beta(time[-1], x, cov)
            return -d_rho * (mu + z0 / rho) / (2 * sigma * sqrt_2_rho)
        return 0.0

    def d2_C(self, i, k, m):
        time = self.time[i]
        x = self.model.x
        cov = self.covariates[i]
        z0 = self.z0
        rho = self.model.f_ro(time[-1], cov)
        sigma = x[0]
        mu = x[1]

        if rho <= 1e-12:
            return 0.0

        sqrt_2_rho = np.sqrt(2 * rho)
        inv_rho = 1.0 / rho
        common_factor = (mu + z0 * inv_rho) / (2 * sigma * sqrt_2_rho)

        d_rho_g = self.model.trend.d_func_gamma(time[-1], x, cov) if k >= 2 or m >= 2 else 0.0
        d_rho_b = self.model.trend.d_func_beta(time[-1], x, cov) if k >= 2 or m >= 2 else 0.0

        def get_d_rho(idx):
            if idx == 2:
                return d_rho_g
            elif idx == 3:
                return d_rho_b
            return 0.0

        def get_d2_rho(idx1, idx2):
            t = time[-1]
            if idx1 == 2 and idx2 == 2:
                return self.model.trend.d2_func_gamma(t, x, cov)
            elif (idx1 == 2 and idx2 == 3) or (idx1 == 3 and idx2 == 2):
                return self.model.trend.d2_func_gamma_beta(t, x, cov)
            elif idx1 == 3 and idx2 == 3:
                return self.model.trend.d2_func_beta(t, x, cov)
            return 0.0

        if k > m:
            k, m = m, k

        if k == 0 and m == 0:
            return 2 * (z0 - mu * rho) / (sigma ** 3 * sqrt_2_rho)
        elif k == 0 and m == 1:
            return np.sqrt(rho / 2) / (sigma ** 2)
        elif k == 0 and m == 2:
            return get_d_rho(2) * common_factor / sigma
        elif k == 0 and m == 3:
            return get_d_rho(3) * common_factor / sigma
        elif k == 1 and m == 1:
            return 0.0
        elif k == 1 and m == 2:
            return -0.5 * get_d_rho(2) / (sigma * sqrt_2_rho)
        elif k == 1 and m == 3:
            return -0.5 * get_d_rho(3) / (sigma * sqrt_2_rho)
        elif k == 2 and m == 2:
            d2 = get_d2_rho(2, 2)
            d1 = get_d_rho(2)
            it = d1**2 * (3*z0 + mu*rho) / (4*sigma*sqrt_2_rho) / rho**2 - d2 * (z0 + mu*rho) / (2*sigma*sqrt_2_rho*rho)
            return it
        elif k == 2 and m == 3:
            d2 = get_d2_rho(2, 3)
            d1g = get_d_rho(2)
            d1b = get_d_rho(3)
            it = d1g*d1b * (3*z0 + mu*rho) / (4*sigma*sqrt_2_rho) / rho**2 - d2 * (z0 + mu*rho) / (2*sigma*sqrt_2_rho*rho)
            return it
        elif k == 3 and m == 3:
            d2 = get_d2_rho(3, 3)
            d1 = get_d_rho(3)
            it = d1**2 * (3*z0 + mu*rho) / (4*sigma*sqrt_2_rho) / rho**2 - d2 * (z0 + mu*rho) / (2*sigma*sqrt_2_rho*rho)
            return it
        return 0.0

    def d_F(self, i, k):
        C_val = self.C(i)
        dC = self.d_C(i, k)
        return dC * np.exp(-C_val ** 2) / np.sqrt(np.pi)

    def d2_F(self, i, k, m):
        C_val = self.C(i)
        dC_k = self.d_C(i, k)
        dC_m = self.d_C(i, m)
        d2C_km = self.d2_C(i, k, m)
        exp_term = np.exp(-C_val ** 2) / np.sqrt(np.pi)
        return (d2C_km - 2 * C_val * dC_k * dC_m) * exp_term

    def d2_lnF(self, k, m):
        total = 0.0
        for i in range(len(self.time)):
            F_val = self.F(i)
            if F_val < 1e-12:
                continue
            dF_k = self.d_F(i, k)
            dF_m = self.d_F(i, m)
            d2F_km = self.d2_F(i, k, m)
            total += (d2F_km / F_val) - (dF_k * dF_m) / (F_val ** 2)
        return float(total)

    def I11_F(self): return self.d2_lnF(0, 0)
    def I12_F(self): return self.d2_lnF(0, 1)
    def I13_F(self): return self.d2_lnF(0, 2)
    def I14_F(self): return self.d2_lnF(0, 3)
    def I22_F(self): return self.d2_lnF(1, 1)
    def I23_F(self): return self.d2_lnF(1, 2)
    def I24_F(self): return self.d2_lnF(1, 3)
    def I33_F(self): return self.d2_lnF(2, 2)
    def I34_F(self): return self.d2_lnF(2, 3)
    def I44_F(self): return self.d2_lnF(3, 3)