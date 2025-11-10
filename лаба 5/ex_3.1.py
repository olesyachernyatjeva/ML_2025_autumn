'''import numpy as np
import matplotlib.pyplot as plt

# === Визуализация ===
def plot_levels(func, xrange=None, yrange=None, levels=None):
    if xrange is None:
        xrange = [-6, 6]
    if yrange is None:
        yrange = [-6, 6]
    if levels is None:
        levels = [0.25, 1, 2, 4, 8, 16, 32]

    x = np.linspace(xrange[0], xrange[1], 200)
    y = np.linspace(yrange[0], yrange[1], 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    CS = plt.contour(X, Y, Z, levels=levels, colors='k', linewidths=0.8)
    plt.clabel(CS, inline=True, fontsize=8)
    plt.grid(alpha=0.4)

def plot_trajectory(func, history, label=None, color='r'):
    x_values, y_values = zip(*history)
    plt.plot(x_values, y_values, '-v', c=color, lw=2, ms=5, label=label)


# === Класс квадратичной функции ===
class QuadOracleCustom:
    """f(x) = 1/2 * x^T A x - b^T x"""
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * x.T @ self.A @ x - self.b.T @ x

    def grad(self, x):
        return self.A @ x - self.b


# === Градиентный спуск ===
def gradient_descent(oracle, x0, alpha=0.1, tol=1e-6, max_iter=5000):
    x = x0.copy()
    history = [x.copy()]
    for _ in range(max_iter):
        grad = oracle.grad(x)
        if np.linalg.norm(grad) < tol:
            break
        x -= alpha * grad
        history.append(x.copy())
    return x, history


# === Основной код ===
if __name__ == "__main__":
    # Пример 1: хорошо обусловленная функция
    A1 = np.array([[2, 0], [0, 1]])
    b1 = np.array([0, 0])
    oracle1 = QuadOracleCustom(A1, b1)

    x0 = np.array([3.0, 2.0])
    x_opt1, history1 = gradient_descent(oracle1, x0, alpha=0.4)

    plt.figure(figsize=(6, 5))
    plot_levels(oracle1.func, xrange=[-4, 4], yrange=[-4, 4])
    plot_trajectory(oracle1.func, history1, label="Хорошо обусловленная", color='r')
    plt.title("Траектория градиентного спуска (хорошая обусловленность)")
    plt.legend()
    plt.show()

    # Пример 2: плохо обусловленная функция
    A2 = np.array([[10, 0], [0, 1]])
    b2 = np.array([0, 0])
    oracle2 = QuadOracleCustom(A2, b2)

    x0 = np.array([3.0, 2.0])
    x_opt2, history2 = gradient_descent(oracle2, x0, alpha=0.1)

    plt.figure(figsize=(6, 5))
    plot_levels(oracle2.func, xrange=[-4, 4], yrange=[-4, 4])
    plot_trajectory(oracle2.func, history2, label="Плохая обусловленность", color='b')
    plt.title("Траектория градиентного спуска (плохая обусловленность)")
    plt.legend()
    plt.show()

    print("Эксперимент завершён успешно!")'''
import numpy as np
import matplotlib.pyplot as plt
from oracles import QuadraticOracle
from plot_trajectory_2d import plot_levels, plot_trajectory
from optimization import gradient_descent, get_line_search_tool

def test_gradient_descent_cases():
    # Определения функций и стратегий
    cases = [
        (np.array([[1,0],[0,1]]), "Хорошо обусловленная"),
        (np.array([[1,0],[0,100]]), "Плохо обусловленная"),
        (np.array([[2,1.5],[1.5,2]]), "С корреляцией")
    ]
    start_points = [np.array([4,4]), np.array([-3,2]), np.array([0.5,-0.5])]
    strategies = [
        {'method':'Constant','c':0.1},
        {'method':'Constant','c':1.0},
        {'method':'Armijo','c1':1e-4,'alpha_0':1.0},
        {'method':'Wolfe','c1':1e-4,'c2':0.9,'alpha_0':1.0}
    ]
    strategy_names = ['Const(0.1)','Const(1.0)','Armijo','Wolfe']

    for A, label in cases:
        oracle = QuadraticOracle(A, np.zeros(A.shape[0]))
        eigvals = np.linalg.eigvals(A)
        cond = max(eigvals)/min(eigvals)
        print(f"\n{label} (число обусловленности {cond:.1f})")
        for sp in start_points:
            plt.figure(figsize=(12,4))
            plt.suptitle(f"{label}, старт {sp}")
            for i, strat in enumerate(strategies):
                _, _, hist = gradient_descent(
                    oracle, sp, tolerance=1e-6, max_iter=1000,
                    line_search_options=strat, trace=True, display=False)
                plt.subplot(1,4,i+1)
                plot_levels(oracle.func)
                if hist and 'x' in hist:
                    plot_trajectory(oracle.func, hist['x'])
                plt.title(strategy_names[i])
            plt.tight_layout()
            plt.show()

test_gradient_descent_cases()
