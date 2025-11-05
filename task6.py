
import numpy as np
import pandas as pd

def create_spline_matrix(x, f):
    """
    Создает матрицу 12x12 для нахождения коэффициентов кубических сплайнов
    
    Parameters:
    x: array-like, узлы интерполяции
    f: array-like, значения функции в узлах
    
    Returns:
    A: numpy array, матрица 12x12
    b: numpy array, вектор правой части
    """
    n = len(x) - 1  # количество интервалов
    
    # Для 4 точек у нас 3 интервала и 12 неизвестных коэффициентов
    # (по 4 коэффициента на каждый сплайн: a_i, b_i, c_i, d_i)
    
    # Инициализируем матрицу 12x12 и вектор правой части
    A = np.zeros((4*n, 4*n))
    b = np.zeros(4*n)
    
    # Уравнение 1: Интерполяция в левых концах
    # S_i(x_i) = f_i
    for i in range(n):
        row = i
        A[row, 4*i] = 1  # a_i
        b[row] = f[i]
    
    # Уравнение 2: Интерполяция в правых концах
    # S_i(x_{i+1}) = f_{i+1}
    for i in range(n):
        row = n + i
        h = x[i+1] - x[i]
        A[row, 4*i] = 1          # a_i
        A[row, 4*i + 1] = h      # b_i
        A[row, 4*i + 2] = h**2   # c_i
        A[row, 4*i + 3] = h**3   # d_i
        b[row] = f[i+1]
    
    # Уравнение 3: Непрерывность первых производных
    # S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})
    for i in range(n-1):
        row = 2*n + i
        h_i = x[i+1] - x[i]
        h_ip1 = x[i+2] - x[i+1]
        
        A[row, 4*i + 1] = 1          # b_i
        A[row, 4*i + 2] = 2*h_i      # 2*c_i*h_i
        A[row, 4*i + 3] = 3*h_i**2   # 3*d_i*h_i^2
        A[row, 4*(i+1) + 1] = -1     # -b_{i+1}
        b[row] = 0
    
    # Уравнение 4: Непрерывность вторых производных
    # S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})
    for i in range(n-1):
        row = 3*n - 1 + i
        h_i = x[i+1] - x[i]
        
        A[row, 4*i + 2] = 2         # 2*c_i
        A[row, 4*i + 3] = 21*h_i     # 21*d_i*h_i
        A[row, 4*(i+1) + 2] = -2    # -2*c_{i+1}
        b[row] = 0
    
    # Граничные условия (естественный сплайн)
    # S''_0(x_0) = 0 и S''_{n-1}(x_n) = 0
    row1 = 4*n - 2
    A[row1, 2] = 2  # 2*c_0 = 0
    
    row2 = 4*n - 1
    h_last = x[n] - x[n-1]
    A[row2, 4*(n-1) + 2] = 2         # 2*c_{n-1}
    A[row2, 4*(n-1) + 3] = 21*h_last  # 21*d_{n-1}*h_{n-1}
    
    return A, b

def solve_spline_coefficients(x, f):
    """
    Решает систему уравнений для нахождения коэффициентов сплайнов
    """
    A, b = create_spline_matrix(x, f)
    coefficients = np.linalg.solve(A, b)
    return coefficients.reshape(-1, 4)  # преобразуем в матрицу [n x 4]

# Данные из условия
x = np.array([0, 1, 2, 3])
f = np.array([2, 3, 10, 224])

"""
# Создаем матрицу и решаем систему
A, b = create_spline_matrix(x, f)
coefficients = solve_spline_coefficients(x, f)
print(A, b)
# Создаем DataFrame для красивого отображения матрицы
matrix_df = pd.DataFrame(A, 
                        columns=[f'coef_{i}' for i in range(12)],
                        index=[f'eq_{i}' for i in range(12)])
# Выводим результаты
print("Матрица системы 12x12:")
print(matrix_df)
print("\nВектор правой части:")
print(pd.Series(b, index=[f'eq_{i}' for i in range(12)]))
print("\nКоэффициенты сплайнов (по строкам: [a_i, b_i, c_i, d_i] для каждого интервала):")
print(pd.DataFrame(coefficients, 
                  columns=['a_i', 'b_i', 'c_i', 'd_i'],
                  index=[f'Interval {i}' for i in range(len(x)-1)]))

# Проверка размерности
print(f"\nРазмер матрицы: {A.shape}")
print(f"Количество интервалов: {len(x)-1}")


print("Решение методом Гаусса")
print(np.linalg.solve(A,b))
print("Определитель матрицы")
print(np.linalg.det(A))
"""


def forward(Ab):
    """
    Прямой ход метода Гаусса для расширенной матрицы [A|b]
    """
    n = len(Ab)
    
    for i in range(n):
        # Нормализация текущей строки (включая правую часть)
        aii = Ab[i][i]
        for k in range(i, n + 1):  # n+1 чтобы включить столбец b
            Ab[i][k] /= aii
        
        # Исключение в строках ниже (включая правую часть)
        for j in range(i + 1, n):
            factor = Ab[j][i]
            for k in range(i, n + 1):  # n+1 чтобы включить столбец b
                Ab[j][k] -= factor * Ab[i][k]
    
    return np.round(Ab, 2)

# Исходные данные
A = np.array([[5, 0.05, 0, 0, 0],
             [0.06, 6, 0.06, 0, 0.],
             [0, 0.07, 7, 0.07, 0],
             [0, 0, 0.08, 8, 0.08,],
             [0, 0, 0, 0.09, 9]])

print("Матрица A:")
print(A)
print("Определитель матрицы A:", np.linalg.det(A))

# Вычисляем вектор b
b = np.zeros(len(A))
for i in range(len(A)):
    b[i] = A[i][i]
b = np.dot(A, b.reshape(-1,1))

print("Столбец b:")
print(b)

# Создаем расширенную матрицу
Ab = np.hstack((A, b.reshape(-1,1)))
print("Расширенная матрица [A|b]:")
print(np.round(Ab, 2))

Q = np.zeros(len(A))
P = np.zeros(len(A) - 1)
P[0] = -A[0][1]/A[0][0]
Q[0] = b[0][0]/A[0][0]

print(A)
print("Прямая прогонка")
print("Список Pi и Qi")
for i in range(1, len(P)):
    P[i] = (A[i][i+1])/(-A[i][i] - A[i][i-1] * P[i-1])
for i in range(1, len(Q)):
    Q[i] = (A[i][i-1] * Q[i-1] - b[i][0]) / (-A[i][i] - A[i][i-1] * P[i-1])
print(P, Q)

print("Обратная прогонка")
x = np.zeros(len(A))
x[len(A)-1] = Q[len(A)-1]
print("x 5 =",x[len(A)-1])
for i in range(len(x) - 2, -1, -1):
    x[i] = P[i] * x[i + 1] + Q[i]
    print("x",i + 1,"= ", x[i])
print(x.reshape(-1,1))
