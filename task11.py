import numpy as np
from scipy import integrate
def rhs_func(x, variant): # правая часть уравнения
    return variant * (4 / 3 * x + 1 / 4 * x**2 + 1 / 5 * x**3)
def build_alpha(size):  # матрица коэффициентов a_ij
    alpha = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            def integrand(t):
                if i == 0:
                    ai = t
                elif i == 1:
                    ai = t**2
                else:
                    ai = t**3
                if j == 0:
                    bj = t
                elif j == 1:
                    bj = t**2
                else:
                    bj = t**3
                return ai * bj
            alpha[i, j], _ = integrate.quad(integrand, 0, 1)
    return alpha
def build_gamma(size, variant): # вектор gamma_i
    gamma = np.zeros(size)
    for i in range(size):
        def integrand(t):
            if i == 0:
                bi = t
            elif i == 1:
                bi = t**2
            else:
                bi = t**3
            return rhs_func(t, variant) * bi
        gamma[i], _ = integrate.quad(integrand, 0, 1)
    return gamma
def gauss_method(A, b):# метод Гаусса
    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    for i in range(n): # прямой ход
        max_row = i + np.argmax(abs(A[i:, i])) # поиск максимального элемента для выбора главного элемента
        if A[max_row, i] == 0:
            raise ValueError("Система не имеет единственного решения (нулевой ведущий элемент).")
        if max_row != i: # Перестановка строк
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        pivot = A[i, i] # нормализация ведущей строки
        A[i] = A[i] / pivot
        b[i] = b[i] / pivot
        for j in range(i + 1, n): # обнуление элементов под ведущим
            factor = A[j, i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]
    x = np.zeros(n) # Обратный ход
    for i in range(n - 1, -1, -1):
        x[i] = b[i] - np.dot(A[i, i + 1:], x[i + 1:])
    return x
def fredholm_solver(variant, rank=3):
    alpha = build_alpha(rank)
    gamma = build_gamma(rank, variant)
    system_matrix = np.eye(rank) + alpha
    coeffs = gauss_method(system_matrix, gamma)
    step = 0.1
    x_vals = np.arange(0, 1 + step, step)
    y_num = np.zeros_like(x_vals)
    for i, x in enumerate(x_vals):
        y_val = rhs_func(x, variant)
        for j in range(rank):
            if j == 0:
                aj = x
            elif j == 1:
                aj = x**2
            else:
                aj = x**3
            y_val -= coeffs[j] * aj
        y_num[i] = y_val
    y_true = variant * x_vals
    err = np.abs(y_num - y_true)
    return x_vals, y_num, y_true, err
def main():
    V = 5
    print(f"Вычисление для варианта {V}")
    x, y_calc, y_exact, error = fredholm_solver(V)
    print("\nРешение интегрального уравнения Фредгольма (вырожденное ядро)")
    print("x:           ", " ".join(f"{float(xi):7.3f}" for xi in x))
    print("y_мет:       ", " ".join(f"{float(yi):7.3f}" for yi in y_calc))
    print("y_точн:      ", " ".join(f"{float(yi):7.3f}" for yi in y_exact))
    print("погрешность: ", " ".join(f"{ei:7.3f}" for ei in error))
if __name__ == "__main__":
    main()