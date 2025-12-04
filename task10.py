import numpy as np

V = 5 # номер варианта

h = 0.1 # шаг
n = int(V / h)
maxx = V

# y точный
def ytoch(x):
    return V * x * x * (x - maxx)

def f(x):
    return 4 * V * (x ** 4) - 3 * V * V * (x ** 3) + 6 * V * x - 2 * V * V

def p(x):
    return x * x

def q(x):
    return x

def phi_k(x, k):
    return x ** k * (x - V)

def dphi_k(x, k):
    return (k + 1) * x ** k - V * k * x ** (k - 1)

def ddphi_k(x, k):
    return k * (k + 1) * x ** (k - 1) - V * k * (k - 1) * x ** (k - 2)

# Создаем сетку
xk = [x * h for x in range(n + 1)]
ykToch = [ytoch(x) for x in xk]

print("Проверка точного решения в ключевых точках:")
test_points = [0, 1, 2, 3, 4, 5]
for x in test_points:
    print(f"x = {x}: y_toch = {ytoch(x):.2f}")

# Инициализация матрицы A и вектора b
A = np.zeros((n, n))
b = np.zeros(n)

# Заполнение матрицы A и вектора b для ВНУТРЕННИХ точек (i=1..n-1)
for i in range(1, n + 1):
    b[i - 1] = f(xk[i])
    for k in range(1, n + 1):
        A[i - 1][k - 1] = ddphi_k(xk[i], k) + p(xk[i]) * dphi_k(xk[i], k) + q(xk[i]) * phi_k(xk[i], k)

print(f"\nРазмерность матрицы A: {A.shape}")
print(f"Размерность вектора b: {b.shape}")

# Метод Гаусса для решения СЛАУ A * c = b
def gauss_elimination(A, b):
    n = len(b)
    
    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск максимального элемента в столбце для улучшения устойчивости
        max_row = i
        max_val = abs(A[i, i])
        for j in range(i + 1, n):
            if abs(A[j, i]) > max_val:
                max_val = abs(A[j, i])
                max_row = j
        
        # Обмен строк, если необходимо
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]
        
        # Проверка на нулевой диагональный элемент
        if abs(A[i, i]) < 1e-12:
            print(f"Внимание: малый диагональный элемент A[{i},{i}] = {A[i, i]}")
            A[i, i] = 1e-12
        
        # Нормировка строки
        pivot = A[i, i]
        for j in range(i, n):
            A[i, j] /= pivot
        b[i] /= pivot
        
        # Исключение переменной из нижележащих строк
        for j in range(i + 1, n):
            factor = A[j, i]
            for k in range(i, n):
                A[j, k] -= factor * A[i, k]
            b[j] -= factor * b[i]
    
    # Обратный ход метода Гаусса
    c = np.zeros(n)
    for i in range(n - 1, -1, -1):
        c[i] = b[i]
        for j in range(i + 1, n):
            c[i] -= A[i, j] * c[j]
    
    return c

# Решение системы
print("\nРешение СЛАУ методом Гаусса")
c = gauss_elimination(A.copy(), b.copy())

print("\nПервые 10 коэффициентов a_k:")
for i in range(min(10, len(c))):
    print(f"a_{i+1} = {c[i]:.6e}")

# Построение приближенного решения
def y_approx(x, c):
    result = 0
    for k in range(1, len(c) + 1):
        result += c[k - 1] * phi_k(x, k)
    return result

# Сравнение точного и приближенного решений
print("\nСравнение решений в некоторых точках:")
print("x\t\tТочное y\tПриближенное y\t\tОтносительная погрешность")
for x in range(V + 1):
    y_exact = ytoch(x)
    y_appr = y_approx(x, c)
    rel_error = abs((y_appr - y_exact) / y_exact) if abs(y_exact) > 1e-12 else abs(y_appr)
    print(f"{x}\t\t{y_exact:.4f}\t\t{y_appr:.4f}\t\t{rel_error:.2e}")
