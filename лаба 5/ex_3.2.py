'''import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

def generate_quadratic_problem(n, k):
    """
    Генерирует случайную квадратичную задачу размера n с числом обусловленности k
    
    f(x) = 1/2 * x^T A x - b^T x
    
    Parameters:
    -----------
    n : int
        Размерность задачи
    k : float
        Число обусловленности (k >= 1)
        
    Returns:
    --------
    A : sparse diagonal matrix
        Матрица квадратичной формы
    b : ndarray
        Вектор линейной части
    """
    # Генерируем диагональные элементы в диапазоне [1, k]
    # Чтобы обеспечить min=1 и max=k, используем логарифмическое распределение
    log_a = np.random.uniform(0, np.log(k), n)
    a = np.exp(log_a)
    a[0] = 1.0  # Минимальное собственное значение
    a[-1] = k   # Максимальное собственное значение
    
    # Перемешиваем диагональные элементы
    np.random.shuffle(a)
    
    # Создаем диагональную матрицу в разреженном формате
    A = sparse.diags(a, format='csr')
    
    # Генерируем случайный вектор b
    b = np.random.randn(n)
    
    return A, b

def gradient_descent(A, b, x0, tolerance=1e-6, max_iter=10000):
    """
    Реализация градиентного спуска для квадратичной задачи
    
    Parameters:
    -----------
    A : sparse matrix
        Матрица квадратичной формы
    b : ndarray
        Вектор линейной части
    x0 : ndarray
        Начальная точка
    tolerance : float
        Требуемая точность
    max_iter : int
        Максимальное число итераций
        
    Returns:
    --------
    x_star : ndarray
        Найденное решение
    iterations : int
        Число выполненных итераций
    history : list
        История значений функции
    """
    x = x0.copy()
    iterations = 0
    
    # Вычисляем градиент: ∇f(x) = A x - b
    gradient = A.dot(x) - b
    grad_norm = np.linalg.norm(gradient)
    
    # Оптимальный шаг для квадратичной функции: 2 / (λ_min + λ_max)
    # Для диагональной матрицы собственные значения - это диагональные элементы
    if hasattr(A, 'diagonal'):
        diag = A.diagonal()
    else:
        diag = A.toarray().diagonal()
    
    lambda_min = np.min(diag)
    lambda_max = np.max(diag)
    step_size = 2.0 / (lambda_min + lambda_max)
    
    while grad_norm > tolerance and iterations < max_iter:
        # Шаг градиентного спуска
        x = x - step_size * gradient
        
        # Обновляем градиент
        gradient = A.dot(x) - b
        grad_norm = np.linalg.norm(gradient)
        iterations += 1
    
    return x, iterations

def run_experiment(n_values, k_values, num_trials=5, tolerance=1e-6):
    """
    Проводит эксперимент по исследованию зависимости T(n,k)
    
    Parameters:
    -----------
    n_values : list
        Список значений размерности
    k_values : ndarray
        Массив значений чисел обусловленности
    num_trials : int
        Число повторений для каждого набора параметров
    tolerance : float
        Требуемая точность
        
    Returns:
    --------
    results : dict
        Словарь с результатами T(n,k) для каждого n
    """
    results = {}
    
    for n in n_values:
        print(f"Проводим эксперимент для n = {n}")
        T_nk = []
        
        for k in k_values:
            T_k = []
            
            for trial in range(num_trials):
                # Генерируем случайную задачу
                A, b = generate_quadratic_problem(n, k)
                
                # Случайная начальная точка
                x0 = np.random.randn(n)
                
                # Запускаем градиентный спуск
                _, iterations = gradient_descent(A, b, x0, tolerance)
                T_k.append(iterations)
            
            T_nk.append(T_k)
        
        results[n] = np.array(T_nk)
    
    return results

def plot_results(results, k_values, n_values):
    """
    Строит графики зависимости T(n,k) от k для разных n
    """
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    plt.figure(figsize=(12, 8))
    
    for i, n in enumerate(n_values):
        color = colors[i % len(colors)]
        T_nk = results[n]
        
        # Рисуем все кривые для данного n одним цветом
        for trial in range(T_nk.shape[1]):
            plt.semilogy(k_values, T_nk[:, trial], 
                        color=color, alpha=0.6, linewidth=1)
        
        # Средняя кривая для данного n
        mean_T = np.mean(T_nk, axis=1)
        plt.semilogy(k_values, mean_T, 
                    color=color, linewidth=3, 
                    label=f'n = {n}', marker='o', markersize=6)
    
    plt.xlabel('Число обусловленности k', fontsize=14)
    plt.ylabel('Число итераций T(n,k)', fontsize=14)
    plt.title('Зависимость числа итераций градиентного спуска\nот числа обусловленности и размерности', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Основная часть программы
if __name__ == "__main__":
    # Параметры эксперимента
    n_values = [10, 100, 1000]  # Размерности по логарифмической сетке
    k_values = np.geomspace(1, 1000, 20)  # Числа обусловленности по логарифмической сетке
    num_trials = 5  # Число повторений для каждого набора параметров
    tolerance = 1e-6  # Требуемая точность
    
    print("Начинаем эксперимент...")
    print(f"Размерности: {n_values}")
    print(f"Числа обусловленности: от {k_values[0]:.1f} до {k_values[-1]:.1f}")
    print(f"Число повторений: {num_trials}")
    print(f"Точность: {tolerance}")
    print()
    
    # Проводим эксперимент
    results = run_experiment(n_values, k_values, num_trials, tolerance)
    
    # Строим графики
    plot_results(results, k_values, n_values)
    
    # Анализ результатов
    print("\nАнализ результатов:")
    print("=" * 50)
    
    for n in n_values:
        T_nk = results[n]
        mean_last = np.mean(T_nk[-1, :])  # Среднее для максимального k
        print(f"n = {n}: среднее число итераций при k = {k_values[-1]:.1f}: {mean_last:.1f}")
    
    print("\nВыводы:")
    print("1. Число итераций сильно зависит от числа обусловленности k")
    print("2. Зависимость T(k) приблизительно линейная в логарифмической шкале")
    print("3. Размерность n слабо влияет на число итераций для фиксированного k")
    print("4. При больших k сходимость значительно замедляется")'''
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Простая реализация квадратичного оракула

class QuadraticOracle:
    """
    f(x) = 0.5 x^T A x - b^T x
    A может быть scipy.sparse.diags (диагональная) или плотной матрицей
    """
    def __init__(self, A, b):
        self.A = A  # sparse or dense
        self.b = np.asarray(b, dtype=float)

    def func(self, x):
        x = np.asarray(x, dtype=float)
        Ax = self.A.dot(x)
        return 0.5 * np.dot(Ax, x) - np.dot(self.b, x)

    def grad(self, x):
        x = np.asarray(x, dtype=float)
        return self.A.dot(x) - self.b

# Простой gradient descent с backtracking Armijo (возвращает число итераций)

def gradient_descent_armijo(oracle, x0, tol=1e-6, max_iter=10000,
                            alpha0=1.0, c1=1e-4, beta=0.5, display=False):
    x = x0.astype(float).copy()
    grad = oracle.grad(x)
    grad_norm = np.linalg.norm(grad)
    it = 0
    start = time.time()
    while grad_norm > tol and it < max_iter:
        d = -grad
        # Armijo backtracking
        alpha = alpha0
        f0 = oracle.func(x)
        g0 = np.dot(grad, d)
        # safety in case g0 is 0 or positive
        if g0 >= 0:
            # not a descent direction, bail out
            return it, False
        while oracle.func(x + alpha * d) > f0 + c1 * alpha * g0:
            alpha *= beta
            if alpha < 1e-16:
                # too small step
                return it, False
        x = x + alpha * d
        grad = oracle.grad(x)
        grad_norm = np.linalg.norm(grad)
        it += 1
        if display and (it % 1000 == 0):
            print(f"it={it}, grad={grad_norm:.3e}, alpha={alpha:.3e}")
    success = grad_norm <= tol
    return it, success

# Генерация диагональной матрицы A с заданным kappa

def generate_diag_problem(n, kappa, random_state=None):
    rng = np.random.default_rng(random_state)
    # сделаем так, чтобы гарантированно min(a)=1, max(a)=kappa
    a = rng.random(n)
    # internal values in (0,1)
    # установим первый элемент 1, второй kappa, остальные в (1,kappa)
    if n == 1:
        diag = np.array([1.0])
    else:
        other = 1.0 + (kappa - 1.0) * rng.random(n - 2)
        diag = np.empty(n, dtype=float)
        diag[0] = 1.0
        diag[1] = kappa
        diag[2:] = other
    A = sp.diags(diag, offsets=0, format='csr')
    b = rng.normal(size=n)
    return A, b, diag

# Эксперимент: для набора n и kappas повторяем 'repeats' раз

def experiment_for_n(n, kappas, repeats=5, tol=1e-6, max_iter=20000,
                     alpha0=1.0, c1=1e-4, beta=0.5, random_seed_base=0):
    results = []  # список из repeats элементов, каждый — dict: kappa->iterations
    for rep in range(repeats):
        rep_res = {}
        for kappa in kappas:
            A, b, diag = generate_diag_problem(n, kappa, random_state=random_seed_base + rep)
            oracle = QuadraticOracle(A, b)
            x0 = np.zeros(n, dtype=float)
            iters, success = gradient_descent_armijo(oracle, x0, tol=tol,
                                                     max_iter=max_iter, alpha0=alpha0,
                                                     c1=c1, beta=beta)
            # если не сошлось, отметим как max_iter (или можно np.nan)
            if not success:
                rep_res[kappa] = max_iter
            else:
                rep_res[kappa] = iters
        results.append(rep_res)
    return results

# Параметры эксперимента

kappas = np.logspace(0, 6, num=13)   # от 1 до 1e6 (13 точек: 1, 10^(0.5), ..., 1e6)
n_list = [10, 100, 1000]             # набор размерностей (можно расширить)
repeats = 6                          # число случайных экземпляров для каждой пары (n, kappa)
tol = 1e-6                           # критерий останова: grad <= tol
max_iter = 20000

# Выполнение эксперимента (может занять немного времени при больших n)

all_results = {}  # n -> list of dicts (each dict maps kappa->iters)
for n in n_list:
    print(f"Running experiments for n = {n} ...")
    res = experiment_for_n(n, kappas, repeats=repeats, tol=tol, max_iter=max_iter,
                           alpha0=1.0, c1=1e-4, beta=0.5, random_seed_base=1000*n)
    all_results[n] = res


# Построение графиков: T(κ,n) vs κ (по оси x — логарифм κ)

plt.figure(figsize=(10, 6))
colors = {n_list[i]: c for i, c in enumerate(['red', 'blue', 'green'])}
for n in n_list:
    res_list = all_results[n]
    for rep_res in res_list:
        # упорядочим kappas
        y = [rep_res[k] for k in kappas]
        plt.plot(kappas, y, color=colors[n], alpha=0.6, marker='o', linestyle='-')
# легенда: по цвету — n
for n in n_list:
    plt.plot([], [], color=colors[n], label=f"n={n}", linewidth=3)
plt.xscale('log')
plt.yscale('log')  # итерации удобно на log-scale (часто растёт быстро)
plt.xlabel(r'Condition number $\kappa$ (log scale)')
plt.ylabel('Iterations until grad <= tol (log scale)')
plt.title(f'Gradient descent (Armijo) iterations T(κ,n) for tol={tol}')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.show()
