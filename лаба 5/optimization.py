import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.optimize import line_search
from datetime import datetime
from collections import defaultdict


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).
        """
        if self._method == 'Constant':
            return self.c
        
        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            
            # Armijo condition: phi(alpha) <= phi(0) + c1 * alpha * phi'(0)
            phi_0 = oracle.func(x_k)
            grad_phi_0 = oracle.grad_directional(x_k, d_k, 0)
            
            while oracle.func(x_k + alpha * d_k) > phi_0 + self.c1 * alpha * grad_phi_0:
                alpha /= 2
                if alpha < 1e-16:  # Avoid infinite loop
                    return alpha
            return alpha
        
        elif self._method == 'Wolfe':
            # Define functions for line_search
            def phi(alpha):
                return oracle.func(x_k + alpha * d_k)
            
            def derphi(alpha):
                return oracle.grad_directional(x_k, d_k, alpha)
            
            # Use scipy's line_search with Wolfe conditions
            try:
                result = line_search(phi, derphi, x_k, d_k, 
                                   gfk=oracle.grad(x_k),
                                   old_fval=oracle.func(x_k),
                                   c1=self.c1, c2=self.c2)
                
                if result[0] is not None:
                    return result[0]
            except:
                pass
            
            # Если scipy line_search не сработал, используем свою реализацию Wolfe
            alpha = self.alpha_0
            phi_0 = oracle.func(x_k)
            derphi_0 = oracle.grad_directional(x_k, d_k, 0)
            
            # Увеличиваем шаг пока выполняются условия Wolfe
            for i in range(50):  # максимум 50 попыток
                phi_alpha = oracle.func(x_k + alpha * d_k)
                derphi_alpha = oracle.grad_directional(x_k, d_k, alpha)
                
                # Проверяем условия Wolfe
                armijo_ok = phi_alpha <= phi_0 + self.c1 * alpha * derphi_0
                curvature_ok = abs(derphi_alpha) <= self.c2 * abs(derphi_0)
                
                if armijo_ok and curvature_ok:
                    return alpha
                
                # Если производная все еще слишком отрицательная, увеличиваем шаг
                if derphi_alpha < 0:
                    alpha *= 2.0
                else:
                    # Иначе уменьшаем шаг
                    alpha *= 0.5
                    
                if alpha < 1e-16 or alpha > 1e16:
                    break
            
            # Если Wolfe не сработал, fallback на Armijo
            armijo_tool = LineSearchTool(method='Armijo', c1=self.c1, alpha_0=self.alpha_0)
            return armijo_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradient descent optimization method.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    # Initial calculations
    grad_0 = oracle.grad(x_0)
    initial_grad_norm_sq = np.linalg.norm(grad_0) ** 2
    tolerance_scaled = tolerance * initial_grad_norm_sq
    
    previous_alpha = None
    start_time = datetime.now()
    
    for k in range(max_iter):
        # Calculate current gradient and function value
        grad_k = oracle.grad(x_k)
        func_k = oracle.func(x_k)
        grad_norm_sq = np.linalg.norm(grad_k) ** 2
        
        # Store history if needed
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(func_k)
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        # Check stopping criterion
        if grad_norm_sq <= tolerance_scaled:
            if display:
                print(f"Iteration {k}: f(x) = {func_k:.6f}, ||grad|| = {np.sqrt(grad_norm_sq):.6f}")
            return x_k, 'success', history
        
        if display:
            print(f"Iteration {k}: f(x) = {func_k:.6f}, ||grad|| = {np.sqrt(grad_norm_sq):.6f}")
        
        # Compute search direction (negative gradient)
        d_k = -grad_k
        
        # Line search
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        
        if alpha_k is None or np.isinf(alpha_k) or np.isnan(alpha_k):
            return x_k, 'computational_error', history
        
        # Update x
        x_k += alpha_k * d_k
        previous_alpha = alpha_k
    
    # Check final gradient after max_iter
    grad_final = oracle.grad(x_k)
    func_final = oracle.func(x_k)
    grad_norm_sq_final = np.linalg.norm(grad_final) ** 2
    
    if trace:
        current_time = (datetime.now() - start_time).total_seconds()
        history['time'].append(current_time)
        history['func'].append(func_final)
        history['grad_norm'].append(np.sqrt(grad_norm_sq_final))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    
    if display:
        print(f"Final iteration: f(x) = {func_final:.6f}, ||grad|| = {np.sqrt(grad_norm_sq_final):.6f}")
    
    if grad_norm_sq_final <= tolerance_scaled:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    # Initial calculations
    grad_0 = oracle.grad(x_0)
    initial_grad_norm_sq = np.linalg.norm(grad_0) ** 2
    tolerance_scaled = tolerance * initial_grad_norm_sq
    
    start_time = datetime.now()
    
    for k in range(max_iter):
        # Calculate current gradient and function value
        grad_k = oracle.grad(x_k)
        func_k = oracle.func(x_k)
        grad_norm_sq = np.linalg.norm(grad_k) ** 2
        
        # Store history if needed
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(func_k)
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        # Check stopping criterion
        if grad_norm_sq <= tolerance_scaled:
            if display:
                print(f"Iteration {k}: f(x) = {func_k:.6f}, ||grad|| = {np.sqrt(grad_norm_sq):.6f}")
            return x_k, 'success', history
        
        if display:
            print(f"Iteration {k}: f(x) = {func_k:.6f}, ||grad|| = {np.sqrt(grad_norm_sq):.6f}")
        
        try:
            # Compute Hessian and Newton direction
            hess_k = oracle.hess(x_k)
            
            # Solve H d = -g using Cholesky decomposition
            L, lower = scipy.linalg.cho_factor(hess_k)
            d_k = scipy.linalg.cho_solve((L, lower), -grad_k)
            
        except (LinAlgError, ValueError) as e:
            if display:
                print(f"Newton direction error: {e}")
            return x_k, 'computational_error', history
        
        # Line search - always start with alpha=1 for Newton
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        
        if alpha_k is None or np.isinf(alpha_k) or np.isnan(alpha_k):
            return x_k, 'computational_error', history
        
        # Update x
        x_k += alpha_k * d_k
    
    # Check final gradient after max_iter
    grad_final = oracle.grad(x_k)
    func_final = oracle.func(x_k)
    grad_norm_sq_final = np.linalg.norm(grad_final) ** 2
    
    if trace:
        current_time = (datetime.now() - start_time).total_seconds()
        history['time'].append(current_time)
        history['func'].append(func_final)
        history['grad_norm'].append(np.sqrt(grad_norm_sq_final))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())
    
    if display:
        print(f"Final iteration: f(x) = {func_final:.6f}, ||grad|| = {np.sqrt(grad_norm_sq_final):.6f}")
    
    if grad_norm_sq_final <= tolerance_scaled:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history
