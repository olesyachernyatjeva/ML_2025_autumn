import numpy as np
import scipy.sparse as sp
from scipy.special import expit

class BaseSmoothOracle(object):
    def func(self, x):
        raise NotImplementedError
    def grad(self, x):
        raise NotImplementedError
    def hess(self, x):
        raise NotImplementedError
    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))
    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))

class QuadraticOracle(BaseSmoothOracle):
    def __init__(self, A, b):
        if sp.issparse(A):
            # проверяем симметрию (A.A вернёт плотный массив)
            if not np.allclose(A.A.T, A.A):
                raise ValueError("A must be symmetric")
            self.A = A.tocsr()
        else:
            if not np.allclose(A, A.T):
                raise ValueError("A must be symmetric")
            self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)

    def func(self, x):
        Ax = self.A.dot(x)
        return 0.5 * float(np.dot(Ax, x)) - float(np.dot(self.b, x))

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

class LogRegL2Oracle(BaseSmoothOracle):
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        """
        matvec_Ax(x) -> A.dot(x)  (shape m)
        matvec_ATx(x) -> A.T.dot(x) (shape n)
        matmat_ATsA(s) -> A.T * diag(s) * A  (returns dense ndarray or array-like)
        b : vector of labels (shape m,)
        regcoef : float
        """
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.asarray(b)
        self.regcoef = float(regcoef)
        self.m = self.b.shape[0]

    def func(self, x):
        z = self.matvec_Ax(x)
        t = - self.b * z
        loss = np.mean(np.logaddexp(0.0, t))
        reg = 0.5 * self.regcoef * float(np.dot(x, x))
        return float(loss + reg)

    def grad(self, x):
        z = self.matvec_Ax(x)
        s = expit(-self.b * z)
        dz = - self.b * s
        grad_part = self.matvec_ATx(dz) / float(self.m)
        return grad_part + self.regcoef * x

    def hess(self, x):
        # возвращаем плотную матрицу (тесты ожидают матрицу/array)
        z = self.matvec_Ax(x)
        s = expit(-self.b * z)
        w = s * (1.0 - s)
        H = self.matmat_ATsA(w) / float(self.m)
        H = np.asarray(H, dtype=float)  # гарантируем ndarray
        H[np.diag_indices_from(H)] += self.regcoef
        return H

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    (ИСПРАВЛЕНО) Оптимизированный оракул с кэшированием 
    для func/grad/hess и directional-методов.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        # Кэш для func/grad/hess
        self._cached_x = None
        self._cached_z = None # z = Ax
        self._cached_s = None # s = expit(-b*z)
        
        # Кэш для directional
        self._cached_x_dir = None
        self._cached_d_dir = None
        self._cached_Ax = None
        self._cached_Ad = None

    def _compute_z(self, x):
        """Кэширует z = Ax и s = expit(-b*z) для func/grad/hess."""
        if self._cached_x is None or not np.array_equal(self._cached_x, x):
            self._cached_x = np.copy(x)
            self._cached_z = self.matvec_Ax(x)
            self._cached_s = expit(-self.b * self._cached_z) 
        return self._cached_z

    def func(self, x):
        z = self._compute_z(x) # Использует кэш
        t = - self.b * z
        loss = np.mean(np.logaddexp(0.0, t))
        reg = 0.5 * self.regcoef * float(np.dot(x, x))
        return float(loss + reg)

    def grad(self, x):
        z = self._compute_z(x) # Использует кэш
        s = self._cached_s     # Использует кэш
        dz = - self.b * s
        grad_part = self.matvec_ATx(dz) / float(self.m)
        return grad_part + self.regcoef * x

    def hess(self, x):
        z = self._compute_z(x) # Использует кэш
        s = self._cached_s     # Использует кэш
        w = s * (1.0 - s)
        H = self.matmat_ATsA(w) / float(self.m)
        H = np.asarray(H, dtype=float) 
        H[np.diag_indices_from(H)] += self.regcoef
        return H

    def _compute_dir_cache(self, x, d):
        """Кэширует Ax и Ad для directional-методов."""
        if (self._cached_x_dir is None or not np.array_equal(self._cached_x_dir, x) or
            self._cached_d_dir is None or not np.array_equal(self._cached_d_dir, d)):
            
            self._cached_x_dir = np.copy(x)
            self._cached_d_dir = np.copy(d)
            
            if self._cached_x is not None and np.array_equal(self._cached_x, x):
                self._cached_Ax = self._cached_z
            else:
                self._cached_Ax = self.matvec_Ax(x)
            
            self._cached_Ad = self.matvec_Ax(d)

    def func_directional(self, x, d, alpha):
        self._compute_dir_cache(x, d) 
        Ax = self._cached_Ax
        Ad = self._cached_Ad
        
        x_new = x + alpha * d
        z = Ax + alpha * Ad
        s = expit(-self.b * z) # (ИСПРАВЛЕНИЕ) Вычисляем 's' здесь для кэша

        # (ИСПРАВЛЕНИЕ) Обновляем главный кэш
        self._cached_x = np.copy(x_new)
        self._cached_z = z
        self._cached_s = s
        
        t = - self.b * z
        loss = np.mean(np.logaddexp(0.0, t))
        reg = 0.5 * self.regcoef * float(np.dot(x_new, x_new))
        return float(loss + reg)

    def grad_directional(self, x, d, alpha):
        self._compute_dir_cache(x, d) 
        Ax = self._cached_Ax
        Ad = self._cached_Ad

        x_new = x + alpha * d
        z = Ax + alpha * Ad
        s = expit(-self.b * z)

        # (ИСПРАВЛЕНИЕ) Обновляем главный кэш
        self._cached_x = np.copy(x_new)
        self._cached_z = z
        self._cached_s = s
        
        term = (- self.b * s) * Ad
        directional_part = float(np.mean(term))
        reg_part = float(self.regcoef * np.dot(x_new, d)) # (ИСПРАВЛЕНИЕ) Должно быть x_new, а не x
        return directional_part + reg_part

def create_log_reg_oracle(A, b, regcoef, oracle_type='usual', counters=None):
    """
    Create oracle. If counters dict is provided, the matvec/matmat functions will increment counters.
    Returns instance of LogRegL2Oracle or LogRegL2OptimizedOracle (depending on oracle_type).
    """
    if counters is None:
        counters = {'Ax': 0, 'ATx': 0, 'ATsA': 0}

    if sp.issparse(A):
        A = A.tocsr()
        def matvec_Ax(x):
            counters['Ax'] += 1
            return A.dot(x)
        def matvec_ATx(x):
            counters['ATx'] += 1
            return A.T.dot(x)
        def matmat_ATsA(s):
            counters['ATsA'] += 1
            s = np.asarray(s).ravel()
            DA = A.multiply(s[:, np.newaxis])  # масштабируем строки
            M = A.T.dot(DA)
            return M.toarray()  # возвращаем плотный массив (тесты этого ждут)
    else:
        A = np.asarray(A)
        def matvec_Ax(x):
            counters['Ax'] += 1
            return A.dot(x)
        def matvec_ATx(x):
            counters['ATx'] += 1
            return A.T.dot(x)
        def matmat_ATsA(s):
            counters['ATsA'] += 1
            s = np.asarray(s).ravel()
            M = A.T.dot(s[:, np.newaxis] * A)
            return M

    if oracle_type == 'usual':
        cls = LogRegL2Oracle
    elif oracle_type == 'optimized':
        cls = LogRegL2OptimizedOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)

    oracle = cls(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
    # Позволим тестам проверить counters (если им нужен этот объект)
    try:
        oracle.counters = counters
    except Exception:
        pass
    return oracle

def grad_finite_diff(func, x, eps=1e-8):
    """
    Forward finite differences:
      result_i := (f(x + eps * e_i) - f(x)) / eps
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    f0 = float(func(x))
    grad = np.zeros(n, dtype=float)
    for i in range(n):
        x_pert = x.copy()
        x_pert[i] += eps
        grad[i] = (float(func(x_pert)) - f0) / eps
    return grad

def hess_finite_diff(func, x, eps=1e-5):
    """
    Second-order finite differences:
      H_{ij} = (f(x+ei+ej) - f(x+ei) - f(x+ej) + f(x)) / eps^2
    Uses symmetry: compute for j>=i and mirror.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)
    f0 = float(func(x))

    # precompute f(x + eps e_i)
    f_ei = np.empty(n, dtype=float)
    for i in range(n):
        x_i = x.copy()
        x_i[i] += eps
        f_ei[i] = float(func(x_i))

    for i in range(n):
        for j in range(i, n):
            x_ij = x.copy()
            x_ij[i] += eps
            x_ij[j] += eps
            f_ei_ej = float(func(x_ij))
            Hij = (f_ei_ej - f_ei[i] - f_ei[j] + f0) / (eps ** 2)
            H[i, j] = Hij
            H[j, i] = Hij
    return H
