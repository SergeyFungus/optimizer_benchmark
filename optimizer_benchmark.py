"""
optimizer_benchmark.py
Авторы: Дудин Сергей Олегович и Дудина Полина Андреевна
"""

import numpy as np
import time
import warnings
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class OptimizationResult:
    """Класс для хранения результатов оптимизации"""
    method: str
    success: bool
    x_final: np.ndarray
    f_final: float
    niter: int
    nfev: int
    ngev: int
    time_elapsed: float
    grad_norm: float
    error_code: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'success': self.success,
            'f_final': float(self.f_final),
            'niter': int(self.niter),
            'nfev': int(self.nfev),
            'ngev': int(self.ngev),
            'time': float(self.time_elapsed),
            'grad_norm': float(self.grad_norm),
            'error_code': self.error_code
        }


class TestFunctions:
    """Набор тестовых функций для бенчмаркинга"""

    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Сферическая функция (выпуклая, хорошо обусловлена)"""
        return np.sum(x**2)

    @staticmethod
    def sphere_grad(x: np.ndarray) -> np.ndarray:
        return 2 * x

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Функция Розенброка (овражная)"""
        return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

    @staticmethod
    def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
        n = len(x)
        grad = np.zeros(n)
        for i in range(n-1):
            grad[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
            grad[i+1] += 200*(x[i+1] - x[i]**2)
        return grad

    @staticmethod
    def himmelblau(x: np.ndarray) -> float:
        """Функция Химмельблау (мультимодальная, 2D)"""
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

    @staticmethod
    def himmelblau_grad(x: np.ndarray) -> np.ndarray:
        grad = np.zeros(2)
        grad[0] = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
        grad[1] = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
        return grad

    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Функция Растригина (мультимодальная, много локальных минимумов)"""
        n = len(x)
        return 10*n + np.sum(x**2 - 10*np.cos(2*np.pi*x))

    @staticmethod
    def rastrigin_grad(x: np.ndarray) -> np.ndarray:
        return 2*x + 20*np.pi*np.sin(2*np.pi*x)

    @staticmethod
    def elliptic(x: np.ndarray) -> float:
        """Эллиптическая функция (плохая обусловленность)"""
        n = len(x)
        return sum(10**(6*i/(n-1)) * x[i]**2 for i in range(n))

    @staticmethod
    def elliptic_grad(x: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.array([2 * 10**(6*i/(n-1)) * x[i] for i in range(n)])

    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Функция Аккли (мультимодальная с осцилляциями)"""
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2*np.pi*x))
        return -20*np.exp(-0.2*np.sqrt(sum_sq/n)) - np.exp(sum_cos/n) + 20 + np.e

    @staticmethod
    def ackley_grad(x: np.ndarray) -> np.ndarray:
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2*np.pi*x))

        if sum_sq < 1e-10:
            term1 = np.zeros(n)
        else:
            term1 = -20 * np.exp(-0.2*np.sqrt(sum_sq/n)) * (-0.2) * (1/np.sqrt(sum_sq/n)) * (2*x/n)
        term2 = -np.exp(sum_cos/n) * (-2*np.pi*np.sin(2*np.pi*x)/n)

        return term1 + term2


class OptimizerWrapper:
    """Унифицированная обёртка для всех методов оптимизации"""

    AVAILABLE_METHODS = [
        'Nelder-Mead', 'Hooke-Jeeves', 'Rosenbrock', 'Powell',
        'CoordinateDescent', 'GradientDescent', 'SteepestDescent',
        'ConjugateGradient', 'Momentum', 'Nesterov', 'BFGS'
    ]

    def __init__(self, func: Callable, grad: Optional[Callable] = None,
                 tol: float = 1e-6, maxiter: int = 5000, maxfev: Optional[int] = None,
                 noise_std: float = 0.0):
        self.func = func
        self.grad = grad
        self.tol = tol
        self.maxiter = maxiter
        self.maxfev = maxfev if maxfev else 100000
        self.noise_std = noise_std
        self.n_evals_f = 0
        self.n_evals_g = 0
        self.start_time = None

    def _add_noise(self, value: float) -> float:
        if self.noise_std > 0:
            return value + np.random.normal(0, self.noise_std)
        return value

    def _wrapper_f(self, x: np.ndarray) -> float:
        self.n_evals_f += 1
        if self.n_evals_f > self.maxfev:
            raise RuntimeError(f"Превышено максимальное число вычислений функции: {self.maxfev}")
        return self._add_noise(self.func(x))

    def _wrapper_g(self, x: np.ndarray) -> np.ndarray:
        self.n_evals_g += 1
        return self.grad(x)

    def _finite_diff_grad(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = len(x)
        grad = np.zeros(n)
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self._wrapper_f(x_plus) - self._wrapper_f(x_minus)) / (2*eps)
        return grad

    def run(self, method: str, x0: np.ndarray, **kwargs) -> OptimizationResult:
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"Метод {method} не поддерживается.")

        self.n_evals_f = 0
        self.n_evals_g = 0
        self.start_time = time.perf_counter()

        try:
            if method == 'BFGS':
                result = self._run_bfgs(x0, **kwargs)
            elif method == 'ConjugateGradient':
                result = self._run_cg(x0, **kwargs)
            elif method == 'Nelder-Mead':
                result = self._run_nelder_mead(x0, **kwargs)
            elif method == 'Powell':
                result = self._run_powell(x0, **kwargs)
            elif method == 'GradientDescent':
                result = self._run_gradient_descent(x0, **kwargs)
            elif method == 'SteepestDescent':
                result = self._run_steepest_descent(x0, **kwargs)
            elif method == 'Momentum':
                result = self._run_momentum(x0, **kwargs)
            elif method == 'Nesterov':
                result = self._run_nesterov(x0, **kwargs)
            elif method == 'CoordinateDescent':
                result = self._run_coordinate_descent(x0, **kwargs)
            elif method == 'Hooke-Jeeves':
                result = self._run_hooke_jeeves(x0, **kwargs)
            elif method == 'Rosenbrock':
                result = self._run_rosenbrock_method(x0, **kwargs)
            else:
                raise ValueError(f"Неизвестный метод: {method}")

            elapsed = time.perf_counter() - self.start_time

            if self.grad:
                grad_norm = np.linalg.norm(self.grad(result.x_final))
            else:
                grad_norm = np.linalg.norm(self._finite_diff_grad(result.x_final))

            return OptimizationResult(
                method=method, success=result.success, x_final=result.x_final,
                f_final=result.f_final, niter=result.niter, nfev=self.n_evals_f,
                ngev=self.n_evals_g, time_elapsed=elapsed, grad_norm=grad_norm
            )

        except Exception as e:
            elapsed = time.perf_counter() - self.start_time
            return OptimizationResult(
                method=method, success=False, x_final=x0, f_final=float('inf'),
                niter=0, nfev=self.n_evals_f, ngev=self.n_evals_g,
                time_elapsed=elapsed, grad_norm=float('inf'), error_code=str(e)
            )

    # ==================== Реализации методов (сокращённо) ====================

    def _run_bfgs(self, x0, **kwargs):
        res = minimize(self._wrapper_f, x0, method='BFGS',
                      jac=self._wrapper_g if self.grad else None,
                      options={'maxiter': self.maxiter, 'gtol': self.tol, 'disp': False})
        return self._scipy_to_result(res)

    def _run_cg(self, x0, **kwargs):
        res = minimize(self._wrapper_f, x0, method='CG',
                      jac=self._wrapper_g if self.grad else None,
                      options={'maxiter': self.maxiter, 'gtol': self.tol, 'disp': False})
        return self._scipy_to_result(res)

    def _run_nelder_mead(self, x0, **kwargs):
        res = minimize(self._wrapper_f, x0, method='Nelder-Mead',
                      options={'maxiter': self.maxiter, 'xatol': self.tol, 'fatol': self.tol, 'disp': False})
        return self._scipy_to_result(res)

    def _run_powell(self, x0, **kwargs):
        res = minimize(self._wrapper_f, x0, method='Powell',
                      options={'maxiter': self.maxiter, 'xtol': self.tol, 'ftol': self.tol, 'disp': False})
        return self._scipy_to_result(res)

    def _run_gradient_descent(self, x0, lr=0.01, **kwargs):
        x = x0.copy()
        niter = 0
        for i in range(self.maxiter):
            grad = self.grad(x) if self.grad else self._finite_diff_grad(x)
            if np.linalg.norm(grad) < self.tol:
                break
            x = x - lr * grad
            self.n_evals_f += 1
            niter += 1
        return self._create_result(x, niter, True)

    def _run_steepest_descent(self, x0, **kwargs):
        x = x0.copy()
        niter = 0
        for i in range(self.maxiter):
            grad = self.grad(x) if self.grad else self._finite_diff_grad(x)
            if np.linalg.norm(grad) < self.tol:
                break

            def line_obj(alpha):
                self.n_evals_f += 1
                return self.func(x - alpha * grad)

            res = minimize_scalar(line_obj, bracket=(0, 1), method='golden')
            x = x - res.x * grad
            niter += 1
        return self._create_result(x, niter, True)

    def _run_momentum(self, x0, lr=0.01, momentum=0.9, **kwargs):
        x = x0.copy()
        v = np.zeros_like(x0)
        niter = 0
        for i in range(self.maxiter):
            grad = self.grad(x) if self.grad else self._finite_diff_grad(x)
            if np.linalg.norm(grad) < self.tol:
                break
            v = momentum * v - lr * grad
            x = x + v
            self.n_evals_f += 1
            niter += 1
        return self._create_result(x, niter, True)

    def _run_nesterov(self, x0, lr=0.01, momentum=0.9, **kwargs):
        x = x0.copy()
        v = np.zeros_like(x0)
        niter = 0
        for i in range(self.maxiter):
            x_lookahead = x + momentum * v
            grad = self.grad(x_lookahead) if self.grad else self._finite_diff_grad(x_lookahead)
            if np.linalg.norm(grad) < self.tol:
                break
            v_new = momentum * v - lr * grad
            x = x + v_new
            v = v_new
            self.n_evals_f += 1
            niter += 1
        return self._create_result(x, niter, True)

    def _run_coordinate_descent(self, x0, **kwargs):
        x = x0.copy()
        n = len(x0)
        niter = 0
        for iteration in range(self.maxiter):
            x_old = x.copy()
            for i in range(n):
                def f_1d(alpha):
                    x_test = x.copy()
                    x_test[i] = alpha
                    self.n_evals_f += 1
                    return self.func(x_test)
                res = minimize_scalar(f_1d, bracket=(x[i]-1, x[i]+1), method='golden')
                x[i] = res.x
            niter += 1
            if np.linalg.norm(x - x_old) < self.tol:
                break
        return self._create_result(x, niter, True)

    def _run_hooke_jeeves(self, x0, delta=1.0, alpha=0.5, **kwargs):
        x = x0.copy()
        n = len(x0)
        niter = 0
        while delta > self.tol and niter < self.maxiter:
            f_base = self._wrapper_f(x)
            x_new = x.copy()
            for i in range(n):
                for direction in [delta, -delta]:
                    x_test = x_new.copy()
                    x_test[i] += direction
                    f_test = self._wrapper_f(x_test)
                    if f_test < f_base:
                        x_new = x_test
                        f_base = f_test
                        break
            if np.linalg.norm(x_new - x) > 0:
                direction = x_new - x
                x_pattern = x_new + direction
                if self._wrapper_f(x_pattern) < f_base:
                    x = x_pattern
                else:
                    x = x_new
            else:
                delta *= alpha
            niter += 1
        return self._create_result(x, niter, True)

    def _run_rosenbrock_method(self, x0, delta=1.0, alpha=1.2, beta=0.5, **kwargs):
        x = x0.copy()
        n = len(x0)
        directions = np.eye(n)
        niter = 0
        for iteration in range(self.maxiter):
            x_start = x.copy()
            for i in range(n):
                d = directions[i]
                x_plus = x + delta * d
                if self._wrapper_f(x_plus) < self._wrapper_f(x):
                    x = x_plus
                    delta *= alpha
                else:
                    x_minus = x - delta * d
                    if self._wrapper_f(x_minus) < self._wrapper_f(x):
                        x = x_minus
                        delta *= alpha
                    else:
                        delta *= beta

            new_directions = np.zeros_like(directions)
            new_directions[0] = x - x_start
            for i in range(1, n):
                new_directions[i] = directions[i-1]

            for i in range(n):
                for j in range(i):
                    if np.dot(new_directions[j], new_directions[j]) > 1e-10:
                        proj = np.dot(new_directions[i], new_directions[j]) / np.dot(new_directions[j], new_directions[j])
                        new_directions[i] -= proj * new_directions[j]
                norm = np.linalg.norm(new_directions[i])
                if norm > 1e-10:
                    new_directions[i] /= norm

            directions = new_directions
            niter += 1
            if np.linalg.norm(x - x_start) < self.tol:
                break
        return self._create_result(x, niter, True)

    def _scipy_to_result(self, res):
        return OptimizationResult(
            method='', success=res.success, x_final=res.x, f_final=res.fun,
            niter=res.nit if hasattr(res, 'nit') else 0,
            nfev=res.nfev if hasattr(res, 'nfev') else 0,
            ngev=res.njev if hasattr(res, 'njev') else 0,
            time_elapsed=0, grad_norm=0
        )

    def _create_result(self, x, niter, success):
        f_val = self._wrapper_f(x)
        return OptimizationResult(
            method='', success=success, x_final=x, f_final=f_val,
            niter=niter, nfev=self.n_evals_f, ngev=self.n_evals_g,
            time_elapsed=0, grad_norm=0
        )


class BenchmarkRunner:
    """Класс для проведения бенчмарков с улучшенной визуализацией"""

    def __init__(self, output_dir: str = 'benchmark_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []

    def run_benchmark(self, methods: List[str], test_functions: List[Dict],
                     n_runs: int = 10, dimensions: List[int] = [2, 10],
                     noise_levels: List[float] = [0.0], random_seed: int = 42) -> pd.DataFrame:

        np.random.seed(random_seed)
        all_results = []

        for func_info in test_functions:
            func = func_info['func']
            grad = func_info.get('grad')
            x0_generator = func_info['x0_generator']
            f_opt = func_info['f_opt']

            for dim in dimensions:
                if func_info['name'] == 'Himmelblau' and dim != 2:
                    continue

                for noise in noise_levels:
                    print(f"\n{'='*60}")
                    print(f"Тест: {func_info['name']}, dim={dim}, noise={noise}")
                    print('='*60)

                    for run_idx in range(n_runs):
                        x0 = x0_generator(dim, run_idx)

                        wrapper = OptimizerWrapper(
                            func=lambda x: func(x), grad=grad, noise_std=noise
                        )

                        for method in methods:
                            result = wrapper.run(method, x0)

                            result_dict = result.to_dict()
                            result_dict.update({
                                'function': func_info['name'],
                                'dimension': dim,
                                'noise': noise,
                                'run_idx': run_idx,
                                'f_opt': f_opt,
                                'error': abs(result.f_final - f_opt)
                            })

                            all_results.append(result_dict)
                            status = "✓" if result.success else "✗"
                            print(f"  {status} {method:20s}: f={result.f_final:.2e}, "
                                  f"nfev={result.nfev}, t={result.time_elapsed:.3f}s")

        self.results = pd.DataFrame(all_results)
        self._save_results()
        return self.results

    def _save_results(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.results.to_csv(self.output_dir / f'results_{timestamp}.csv', index=False)
        print(f"\nРезультаты сохранены в {self.output_dir / f'results_{timestamp}.csv'}")

    def plot_comparison_bars_separate(self, output_dir: str = 'chapter4_results'):
        """
        4 ОТДЕЛЬНЫХ ГРАФИКА: эффективность, время, надёжность, точность
        """
        if self.results.empty:
            print("Нет данных!")
            return

        success_results = self.results[self.results['success'] == True]
        if success_results.empty:
            print("Нет успешных результатов!")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Вычислительная эффективность
        print("\n Сохранение: Вычислительная эффективность...")
        fig, ax = plt.subplots(figsize=(10, 6))
        nfev_data = success_results.groupby('method')['nfev'].mean().sort_values()
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(nfev_data)))
        bars = ax.bar(range(len(nfev_data)), nfev_data.values, color=colors)
        ax.set_xticks(range(len(nfev_data)))
        ax.set_xticklabels(nfev_data.index, rotation=45, ha='right')
        ax.set_ylabel('Среднее число вычислений функции', fontsize=12)
        ax.set_title('Вычислительная эффективность (меньше = лучше)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, nfev_data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.0f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        save_file = output_dir / '01_computational_efficiency.png'
        fig.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {save_file}")
        plt.close()

        # 2. Быстродействие
        print("\n📊 Сохранение: Быстродействие...")
        fig, ax = plt.subplots(figsize=(10, 6))
        time_data = success_results.groupby('method')['time'].mean().sort_values()
        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(time_data)))
        bars = ax.bar(range(len(time_data)), time_data.values, color=colors)
        ax.set_xticks(range(len(time_data)))
        ax.set_xticklabels(time_data.index, rotation=45, ha='right')
        ax.set_ylabel('Среднее время (с)', fontsize=12)
        ax.set_title('Быстродействие (меньше = лучше)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, time_data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        save_file = output_dir / '02_execution_time.png'
        fig.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {save_file}")
        plt.close()

        # 3. Надёжность
        print("\n📊 Сохранение: Надёжность методов...")
        fig, ax = plt.subplots(figsize=(10, 6))
        success_data = self.results.groupby('method')['success'].mean().sort_values(ascending=False)
        colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(success_data)))
        bars = ax.bar(range(len(success_data)), success_data.values, color=colors)
        ax.set_xticks(range(len(success_data)))
        ax.set_xticklabels(success_data.index, rotation=45, ha='right')
        ax.set_ylabel('Доля успешных запусков', fontsize=12)
        ax.set_title('Надёжность методов (больше = лучше)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, success_data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1%}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        save_file = output_dir / '03_reliability.png'
        fig.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {save_file}")
        plt.close()

        # 4. Точность
        print("\n📊 Сохранение: Точность решений...")
        fig, ax = plt.subplots(figsize=(10, 6))
        error_data = success_results.groupby('method')['error'].median().sort_values()
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(error_data)))
        bars = ax.bar(range(len(error_data)), error_data.values, color=colors)
        ax.set_xticks(range(len(error_data)))
        ax.set_xticklabels(error_data.index, rotation=45, ha='right')
        ax.set_ylabel('Медианная ошибка', fontsize=12)
        ax.set_title('Точность решений (меньше = лучше)', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, error_data.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1e}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        save_file = output_dir / '04_accuracy.png'
        fig.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {save_file}")
        plt.close()

    def load_results_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Загрузка результатов из существующего CSV файла
        """
        print(f"\n Загрузка результатов из: {csv_path}")
        self.results = pd.read_csv(csv_path)
        print(f"✅ Загружено {len(self.results)} записей")
        print(f"Методы: {self.results['method'].unique()}")
        print(f"Функции: {self.results['function'].unique()}")
        return self.results

    def plot_error_distribution_separate(self, output_dir: str = 'chapter4_results'):
        """
        6 ОТДЕЛЬНЫХ ГРАФИКОВ: по одному для каждой тестовой функции
        """
        if self.results.empty:
            print("Нет данных!")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_functions = sorted(self.results['function'].unique())

        # Фиксированный порядок всех 11 методов
        fixed_method_order = [
            'Nelder-Mead', 'Hooke-Jeeves', 'Rosenbrock', 'Powell',
            'CoordinateDescent', 'GradientDescent', 'SteepestDescent',
            'ConjugateGradient', 'Momentum', 'Nesterov', 'BFGS'
        ]

        method_colors = plt.cm.tab20(np.linspace(0, 1, len(fixed_method_order)))
        color_dict = {method: method_colors[i] for i, method in enumerate(fixed_method_order)}

        for idx, func_name in enumerate(all_functions):
            print(f"\n📊 Сохранение: Распределение ошибок для {func_name}...")

            func_data = self.results[self.results['function'] == func_name].copy()

            # Создаём DataFrame с правильным порядком
            plot_data = []
            for method in fixed_method_order:
                method_errors = func_data[func_data['method'] == method]['error'].values
                if len(method_errors) > 0:
                    for err in method_errors:
                        plot_data.append({'method': method, 'error': err})

            if not plot_data:
                print(f"  ⚠️  Нет данных для {func_name}")
                continue

            plot_df = pd.DataFrame(plot_data)
            plot_df['method'] = pd.Categorical(plot_df['method'],
                                               categories=fixed_method_order,
                                               ordered=True)

            fig, ax = plt.subplots(figsize=(16, 8))
            fig.suptitle(f'Распределение ошибок для функции {func_name}',
                         fontsize=16, fontweight='bold', y=1.02)

            # Используем seaborn boxplot - он сохраняет порядок категорий
            import seaborn as sns
            bp = sns.boxplot(data=plot_df, x='method', y='error',
                             ax=ax, order=fixed_method_order,
                             palette=[color_dict[m] for m in fixed_method_order])

            # Настройка осей
            ax.set_ylabel('Ошибка оптимизации', fontsize=13)
            ax.set_xlabel('Метод оптимизации', fontsize=13)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            ax.axhline(y=1e-6, color='r', linestyle='--', alpha=0.7,
                       label='Порог точности (1e-6)', linewidth=2)
            ax.legend(loc='upper right', fontsize=11)

            # Поворот подписей
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)

            plt.tight_layout()

            save_file = output_dir / f'05_{idx + 1}_error_{func_name.lower()}.png'
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ Сохранено: {save_file}")
            plt.close()

        # Таблица лучших методов
        print("\n" + "=" * 80)
        print("ЛУЧШИЕ МЕТОДЫ ДЛЯ КАЖДОЙ ФУНКЦИИ (по медианной ошибке)")
        print("=" * 80)

        for func_name in all_functions:
            func_data = self.results[self.results['function'] == func_name]
            success_data = func_data[func_data['success'] == True]

            if success_data.empty:
                continue

            median_errors = success_data.groupby('method')['error'].median().sort_values()
            print(f"\n{func_name}:")
            for i, (method, error) in enumerate(median_errors.head(5).items(), 1):
                print(f"  {i}. {method:20s}: {error:.2e}")

    def plot_dimension_scaling_separate(self, output_dir: str = 'chapter4_results'):
        """
        МАСШТАБИРОВАНИЕ И УСТОЙЧИВОСТЬ С ФИКСИРОВАННЫМИ ЦВЕТАМИ
        """
        if self.results.empty or 'dimension' not in self.results.columns:
            print("Нет данных или отсутствует колонка 'dimension'!")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 🔵 ЕДИНАЯ ЦВЕТОВАЯ СХЕМА (такая же, как в профилях Долана-Морé)
        method_colors = {
            'Nelder-Mead': '#1f77b4',  # синий
            'Hooke-Jeeves': '#ff7f0e',  # оранжевый
            'Rosenbrock': '#2ca02c',  # зелёный
            'Powell': '#d62728',  # красный
            'CoordinateDescent': '#9467bd',  # фиолетовый
            'GradientDescent': '#8c564b',  # коричневый
            'SteepestDescent': '#e377c2',  # розовый
            'ConjugateGradient': '#7f7f7f',  # серый
            'Momentum': '#bcbd22',  # жёлтый
            'Nesterov': '#17becf',  # бирюзовый
            'BFGS': '#aec7e8'  # голубой
        }

        fixed_order = list(method_colors.keys())
        dim_data = self.results[self.results['success'] == True].copy()

        # ==================== ГРАФИК 1: Масштабируемость (nfev) ====================
        print("\n📊 Сохранение: Масштабируемость методов...")
        fig1, ax1 = plt.subplots(figsize=(12, 7))

        for method in fixed_order:
            method_data = dim_data[dim_data['method'] == method]
            if method_data.empty:
                continue

            grouped = method_data.groupby('dimension')['nfev'].mean()
            ax1.plot(grouped.index, grouped.values, 'o-', linewidth=2.5,
                     label=method, markersize=10, markeredgewidth=1.5,
                     markeredgecolor='white', color=method_colors[method])

        ax1.set_xlabel('Размерность задачи', fontsize=12)
        ax1.set_ylabel('Среднее число вычислений функции', fontsize=12)
        ax1.set_title('Масштабируемость методов по размерности', fontsize=14, fontweight='bold')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc='best')

        plt.tight_layout()
        save_file1 = output_dir / '07_dimension_scalability.png'
        fig1.savefig(save_file1, dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {save_file1}")
        plt.close()

        # ==================== ГРАФИК 2: Устойчивость (успешность) ====================
        print("\n📊 Сохранение: Устойчивость к размерности...")
        fig2, ax2 = plt.subplots(figsize=(12, 7))

        for method in fixed_order:
            method_data = self.results[self.results['method'] == method]
            if method_data.empty:
                continue

            grouped = method_data.groupby('dimension')['success'].mean()
            ax2.plot(grouped.index, grouped.values, 's-', linewidth=2.5,
                     label=method, markersize=10, markeredgewidth=1.5,
                     markeredgecolor='white', color=method_colors[method])

        ax2.set_xlabel('Размерность задачи', fontsize=12)
        ax2.set_ylabel('Доля успешных запусков', fontsize=12)
        ax2.set_title('Устойчивость к увеличению размерности', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10, loc='best')

        plt.tight_layout()
        save_file2 = output_dir / '08_dimension_stability.png'
        fig2.savefig(save_file2, dpi=300, bbox_inches='tight')
        print(f"✅ Сохранено: {save_file2}")
        plt.close()

    def generate_summary_table(self) -> pd.DataFrame:
        """Генерация сводной таблицы"""
        if self.results.empty:
            return pd.DataFrame()

        summary = self.results.groupby('method').agg({
            'success': ['mean', 'count'],
            'nfev': 'mean',
            'time': 'mean',
            'error': 'median'
        }).round(4)

        return summary

    #-------------------------------------------------------------------------------
    def plot_dolan_more_profiles(self, metric: str = 'nfev', save_path: Optional[str] = None):
        """
        ПРОФИЛИ ЭФФЕКТИВНОСТИ ДОЛАНА-МОРЕ С ФИКСИРОВАННЫМИ ЦВЕТАМИ
        """
        if self.results.empty:
            print("Нет данных для построения профилей!")
            return

        print(f"\n📊 Построение профилей эффективности Долана-Морé (метрика: {metric})...")

        success_results = self.results[self.results['success'] == True].copy()
        if success_results.empty:
            print("Нет успешных результатов!")
            return

        grouped = success_results.groupby(['function', 'dimension'])
        methods = success_results['method'].unique()

        # ФИКСИРОВАННЫЕ ЦВЕТА ДЛЯ КАЖДОГО МЕТОДА
        method_colors = {
            'Nelder-Mead': '#1f77b4',  # синий
            'Hooke-Jeeves': '#ff7f0e',  # оранжевый
            'Rosenbrock': '#2ca02c',  # зелёный
            'Powell': '#d62728',  # красный
            'CoordinateDescent': '#9467bd',  # фиолетовый
            'GradientDescent': '#8c564b',  # коричневый
            'SteepestDescent': '#e377c2',  # розовый
            'ConjugateGradient': '#7f7f7f',  # серый
            'Momentum': '#bcbd22',  # жёлтый
            'Nesterov': '#17becf',  # бирюзовый
            'BFGS': '#aec7e8'  # голубой
        }

        fig, ax = plt.subplots(figsize=(12, 8))

        tau_values = np.logspace(0, 2, 100)

        # Рисуем методы в фиксированном порядке
        fixed_method_order = [
            'Nelder-Mead', 'Hooke-Jeeves', 'Rosenbrock', 'Powell',
            'CoordinateDescent', 'GradientDescent', 'SteepestDescent',
            'ConjugateGradient', 'Momentum', 'Nesterov', 'BFGS'
        ]

        # Только методы, которые есть в данных
        methods_to_plot = [m for m in fixed_method_order if m in methods]

        for method in methods_to_plot:
            method_data = success_results[success_results['method'] == method]
            rho_values = []

            for tau in tau_values:
                solved_count = 0
                total_tasks = 0

                for (func, dim), group in grouped:
                    total_tasks += 1
                    best_value = group[metric].min()

                    method_group = method_data[(method_data['function'] == func) &
                                               (method_data['dimension'] == dim)]

                    if len(method_group) > 0 and best_value > 0:
                        method_value = method_group[metric].min()
                        if method_value <= tau * best_value:
                            solved_count += 1

                if total_tasks > 0:
                    rho_values.append(solved_count / total_tasks)
                else:
                    rho_values.append(0.0)

            # Получаем цвет из словаря
            color = method_colors.get(method, 'black')

            ax.semilogx(tau_values, rho_values, label=method, linewidth=2.5, color=color)

        ax.set_xlabel(r'$\tau$ (коэффициент производительности)', fontsize=13)
        ax.set_ylabel(r'$\rho(\tau)$ (доля решённых задач)', fontsize=13)
        ax.set_title('Профили эффективности Долана-Морé\n' +
                     f'(метрика: {metric})',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim([1, 100])
        ax.set_ylim([0, 1.05])

        ax.text(0.02, 0.98,
                'Выше и левее = лучше\n' +
                'τ=1: доля задач, где метод лучший\n' +
                'τ→: общая доля решённых задач',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ Профиль сохранён: {save_file}")

        plt.show()

    def plot_all_dolan_more_profiles(self, output_dir: str = 'chapter4_results'):
        """
        Строит профили для всех метрик (nfev, time, error)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Профиль по числу вычислений
        self.plot_dolan_more_profiles(metric='nfev',
                                      save_path=output_dir / '09_profile_nfev.png')

        # Профиль по времени
        self.plot_dolan_more_profiles(metric='time',
                                      save_path=output_dir / '10_profile_time.png')

        # Профиль по точности
        self.plot_dolan_more_profiles(metric='error',
                                      save_path=output_dir / '11_profile_error.png')

    def plot_noise_sensitivity(self, output_dir='chapter4_results'):
        """Влияние шума на успешность методов"""
        if self.results.empty or 'noise' not in self.results.columns:
            return

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 7))

        for method in self.results['method'].unique():
            method_data = self.results[self.results['method'] == method]
            grouped = method_data.groupby('noise')['success'].mean()
            ax.plot(grouped.index, grouped.values, 'o-', linewidth=2, label=method)

        ax.set_xlabel('Уровень шума σ', fontsize=12)
        ax.set_ylabel('Доля успешных запусков', fontsize=12)
        ax.set_title('Устойчивость методов к зашумлённости', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        plt.tight_layout()
        save_file = Path(output_dir) / '12_noise_sensitivity.png'
        fig.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.show()

# ==================== ЗАПУСК БЕНЧМАРКА ====================

if __name__ == "__main__":
    print("="*80)
    print("БЕНЧМАРК МЕТОДОВ ОПТИМИЗАЦИИ")
    print("="*80)

    # ВСЕ 6 ТЕСТОВЫХ ФУНКЦИЙ
    test_functions = [
        {
            'name': 'Sphere',
            'func': TestFunctions.sphere,
            'grad': TestFunctions.sphere_grad,
            'x0_generator': lambda dim, idx: np.random.uniform(-5, 5, dim),
            'f_opt': 0.0
        },
        {
            'name': 'Rosenbrock',
            'func': TestFunctions.rosenbrock,
            'grad': TestFunctions.rosenbrock_grad,
            'x0_generator': lambda dim, idx: np.random.uniform(-3, 3, dim),
            'f_opt': 0.0
        },
        {
            'name': 'Himmelblau',
            'func': TestFunctions.himmelblau,
            'grad': TestFunctions.himmelblau_grad,
            'x0_generator': lambda dim, idx: np.random.uniform(-5, 5, 2),
            'f_opt': 0.0
        },
        {
            'name': 'Rastrigin',
            'func': TestFunctions.rastrigin,
            'grad': TestFunctions.rastrigin_grad,
            'x0_generator': lambda dim, idx: np.random.uniform(-5.12, 5.12, dim),
            'f_opt': 0.0
        },
        {
            'name': 'Elliptic',
            'func': TestFunctions.elliptic,
            'grad': TestFunctions.elliptic_grad,
            'x0_generator': lambda dim, idx: np.random.uniform(-5, 5, dim),
            'f_opt': 0.0
        },
        {
            'name': 'Ackley',
            'func': TestFunctions.ackley,
            'grad': TestFunctions.ackley_grad,
            'x0_generator': lambda dim, idx: np.random.uniform(-5, 5, dim),
            'f_opt': 0.0
        }
    ]

    # ВСЕ 11 МЕТОДОВ
    methods = [
        'Nelder-Mead',        # Безградиентный
        'Hooke-Jeeves',       # Безградиентный
        'Rosenbrock',         # Безградиентный
        'Powell',             # Безградиентный
        'CoordinateDescent',  # Безградиентный
        'GradientDescent',    # Градиентный
        'SteepestDescent',    # Градиентный
        'ConjugateGradient',  # Градиентный
        'Momentum',           # Градиентный с моментумом
        'Nesterov',           # Ускоренный Нестерова
        'BFGS'                # Квазиньютоновский
    ]

    # Запуск
    runner = BenchmarkRunner(output_dir='chapter4_results')

    # Вариант 1: Загрузка из существующего CSV
    csv_file = 'chapter4_results/results_20260527_020337.csv'
    try:
        results_df = runner.load_results_from_csv(csv_file)
    except FileNotFoundError:
        print(f"Файл {csv_file} не найден. Запускаю бенчмарк заново...")
        results_df = runner.run_benchmark(
            methods=methods,
            test_functions=test_functions,
            n_runs=30,  # Число запусков для статистики
            dimensions=[2, 10, 50],  # Размерности
            noise_levels=[0.0],  #, 0.01, 0.05, 0.1],   # Тестирование с разным уровнем шума
            random_seed=42
        )

    # Вывод сводной таблицы
    print("\n" + "="*80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    summary = runner.generate_summary_table()
    print(summary)

    # Построение графиков
    print("\n" + "="*80)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ")
    print("="*80)

    # 4 отдельных графика сравнения
    runner.plot_comparison_bars_separate(output_dir='chapter4_results')

    # 6 отдельных графиков распределения ошибок
    runner.plot_error_distribution_separate(output_dir='chapter4_results')

    # 2 отдельных графика масштабирования
    runner.plot_dimension_scaling_separate(output_dir='chapter4_results')

    # Профили Долана-Морé
    print("\n" + "=" * 80)
    print("ПРОФИЛИ ЭФФЕКТИВНОСТИ ДОЛАНА-МОРЕ")
    print("=" * 80)
    runner.plot_all_dolan_more_profiles(output_dir='chapter4_results')

    #runner.plot_noise_sensitivity(output_dir='chapter4_results')

    print("\n" + "="*80)
    print("БЕНЧМАРК ЗАВЕРШЁН")
    print("="*80)
    print(f"Все результаты сохранены в папке: chapter4_results/")
