import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix, issparse
import time
import os
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

# Создаем папки для результатов
os.makedirs('results', exist_ok=True)

class LogisticRegressionOracle:
    def __init__(self, A, y, lam=1.0):
        self.A = A
        self.y = y
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.lam = lam
        
    def func(self, x):
        if issparse(self.A):
            Ax = self.A.dot(x)
        else:
            Ax = self.A @ x
        losses = np.logaddexp(0, -self.y * Ax)
        return np.mean(losses) + 0.5 * self.lam * np.dot(x, x)
    
    def grad(self, x):
        if issparse(self.A):
            Ax = self.A.dot(x)
        else:
            Ax = self.A @ x
        sigmoid = 1.0 / (1.0 + np.exp(self.y * Ax))
        weights = -self.y * sigmoid / self.m
        if issparse(self.A):
            grad = self.A.T.dot(weights) + self.lam * x
        else:
            grad = self.A.T @ weights + self.lam * x
        return grad
    
    def hessian(self, x):
        if issparse(self.A):
            Ax = self.A.dot(x)
        else:
            Ax = self.A @ x
        sigmoid = 1.0 / (1.0 + np.exp(self.y * Ax))
        weights = sigmoid * (1 - sigmoid) / self.m
        if issparse(self.A):
            W = csr_matrix((weights, (range(len(weights)), range(len(weights)))))
            hess = self.A.T.dot(W.dot(self.A)) + self.lam * csr_matrix(np.eye(self.n))
        else:
            W = np.diag(weights)
            hess = self.A.T @ W @ self.A + self.lam * np.eye(self.n)
        return hess

def gradient_descent(oracle, x0, tol=1e-6, max_iter=1000, alpha=0.1):
    x = x0.copy()
    history = {'time': [0.0], 'func': [oracle.func(x)], 'grad_norm': [np.linalg.norm(oracle.grad(x))]}
    grad0_norm = history['grad_norm'][0]
    start_time = time.time()
    
    for k in range(max_iter):
        grad = oracle.grad(x)
        x = x - alpha * grad
        
        current_time = time.time() - start_time
        history['time'].append(current_time)
        history['func'].append(oracle.func(x))
        history['grad_norm'].append(np.linalg.norm(oracle.grad(x)))
        
        if history['grad_norm'][-1] / grad0_norm < tol:
            break
            
    return x, history

def newton_method(oracle, x0, tol=1e-6, max_iter=50):
    x = x0.copy()
    history = {'time': [0.0], 'func': [oracle.func(x)], 'grad_norm': [np.linalg.norm(oracle.grad(x))]}
    grad0_norm = history['grad_norm'][0]
    start_time = time.time()
    
    for k in range(max_iter):
        grad = oracle.grad(x)
        hess = oracle.hessian(x)
        
        try:
            if issparse(hess):
                dx = spsolve(hess, -grad)
            else:
                dx = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:
            if issparse(hess):
                dx = -grad / np.diag(hess.toarray())
            else:
                dx = -grad / np.diag(hess)
            
        x = x + dx
        
        current_time = time.time() - start_time
        history['time'].append(current_time)
        history['func'].append(oracle.func(x))
        history['grad_norm'].append(np.linalg.norm(oracle.grad(x)))
        
        if history['grad_norm'][-1] / grad0_norm < tol:
            break
            
    return x, history

def load_dataset(dataset_name):
    possible_paths = [
        f'data/{dataset_name}',
        f'data/{dataset_name}.txt',
        f'{dataset_name}',
        f'{dataset_name}.txt'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Найден файл: {path}")
            A, y = load_svmlight_file(path)
            return A, y
    
    raise FileNotFoundError(f"Не найден файл для датасета {dataset_name}")

def plot_results(dataset_name, history_gd, history_newton, m, n):
    """Построение графиков в точности как на фото"""
    # Устанавливаем стиль как на фото
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # === ГРАФИК 1: Сходимость по функции ===
    ax1.plot(history_gd['time'], history_gd['func'], 'b-', linewidth=2.5, label='Градиентный спуск')
    ax1.plot(history_newton['time'], history_newton['func'], 'r-', linewidth=2.5, label='Ньютон (CG)')
    
    ax1.set_xlabel('Время (сек)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('f(x)', fontsize=14, fontweight='bold')
    ax1.set_title('Сходимость по функции', fontsize=16, fontweight='bold', pad=20)
    
    # Настройки осей как на фото
    ax1.set_xlim(0, max(history_gd['time'][-1], history_newton['time'][-1]) * 1.05)
    ax1.set_ylim(min(min(history_gd['func']), min(history_newton['func'])) * 0.95, 
                max(history_gd['func'][0], history_newton['func'][0]) * 1.05)
    
    ax1.legend(fontsize=12, framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # === ГРАФИК 2: Сходимость по градиенту (лог) ===
    grad0_norm = history_gd['grad_norm'][0]
    
    # Вычисляем относительную норму градиента в квадрате
    rel_grad_gd = [(g / grad0_norm) ** 2 for g in history_gd['grad_norm']]
    rel_grad_newton = [(g / grad0_norm) ** 2 for g in history_newton['grad_norm']]
    
    ax2.semilogy(history_gd['time'], rel_grad_gd, 'b-', linewidth=2.5, label='Градиентный спуск')
    ax2.semilogy(history_newton['time'], rel_grad_newton, 'r-', linewidth=2.5, label='Ньютон (CG)')
    
    ax2.set_xlabel('Время (сек)', fontsize=14, fontweight='bold')
    ax2.set_ylabel(r'$\|\nabla f(x_k)\|^2 / \|\nabla f(x_0)\|^2$', 
                  fontsize=14, fontweight='bold')
    ax2.set_title('Сходимость по градиенту (лог)', fontsize=16, fontweight='bold', pad=20)
    
    # Настройки для логарифмической шкалы
    ax2.set_ylim(1e-12, 1)
    ax2.set_xlim(0, max(history_gd['time'][-1], history_newton['time'][-1]) * 1.05)
    
    ax2.legend(fontsize=12, framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Общие настройки
    plt.suptitle(f'Датасет: {dataset_name} (m={m}, n={n})', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    plt.savefig(f'results/{dataset_name}_comparison.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

def run_experiment(dataset_name):
    print(f"\n{'='*60}")
    print(f"ЭКСПЕРИМЕНТ: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        A, y = load_dataset(dataset_name)
        
        unique_labels = np.unique(y)
        if set(unique_labels) == {0, 1}:
            y = 2 * y - 1
            print("Метки преобразованы из {0,1} в {-1,1}")
        
        m, n = A.shape
        lam = 1.0 / m
        
        print(f"Размерность: {m} samples, {n} features")
        print(f"Коэффициент регуляризации: λ = {lam:.6f}")
        
        oracle = LogisticRegressionOracle(A, y, lam)
        x0 = np.zeros(n)
        
        print(f"Начальное значение функции: {oracle.func(x0):.6f}")
        
        print("\n--- Запуск градиентного спуска ---")
        x_gd, history_gd = gradient_descent(oracle, x0, max_iter=500, alpha=0.1)
        
        print("\n--- Запуск метода Ньютона ---")
        x_newton, history_newton = newton_method(oracle, x0, max_iter=20)
        
        plot_results(dataset_name, history_gd, history_newton, m, n)
        
        print(f"\n--- РЕЗУЛЬТАТЫ {dataset_name} ---")
        print(f"Градиентный спуск: {len(history_gd['time'])-1} итераций, время: {history_gd['time'][-1]:.2f}с")
        print(f"Метод Ньютона: {len(history_newton['time'])-1} итераций, время: {history_newton['time'][-1]:.2f}с")
        print(f"Лучшее f(x) GD: {min(history_gd['func']):.6f}")
        print(f"Лучшее f(x) Newton: {min(history_newton['func']):.6f}")
        
        return history_gd, history_newton
        
    except Exception as e:
        print(f"Ошибка при обработке {dataset_name}: {e}")
        return None, None

def main():
    datasets = ['w8a', 'gisette', 'real-sim']
    
    for dataset in datasets:
        try:
            run_experiment(dataset)
        except Exception as e:
            print(f"Не удалось обработать {dataset}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ!")
    print("Результаты в папке 'results/'")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
