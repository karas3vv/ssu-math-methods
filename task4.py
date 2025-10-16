
import numpy as np
import pandas as pd

def create_spline_matrix(x, f):
    """
    создает матрицу 12x12 для нахождения коэффициентов кубических сплайнов
    
    parameters:
    x: array-like, узлы интерполяции
    f: array-like, значения функции в узлах
    
    returns:
    a: numpy array, матрица 12x12
    b: numpy array, вектор правой части
    """
    n = len(x) - 1  # количество интервалов
    
    # для 4 точек у нас 3 интервала и 12 неизвестных коэффициентов
    # (по 4 коэффициента на каждый сплайн: a_i, b_i, c_i, d_i)
    
    # инициализируем матрицу 12x12 и вектор правой части
    a = np.zeros((4*n, 4*n))
    b = np.zeros(4*n)
    
    # уравнение 1: интерполяция в левых концах
    # s_i(x_i) = f_i
    for i in range(n):
        row = i
        a[row, 4*i] = 1  # a_i
        b[row] = f[i]
    
    # уравнение 2: интерполяция в правых концах
    # s_i(x_{i+1}) = f_{i+1}
    for i in range(n):
        row = n + i
        h = x[i+1] - x[i]
        a[row, 4*i] = 1          # a_i
        a[row, 4*i + 1] = h      # b_i
        a[row, 4*i + 2] = h**2   # c_i
        a[row, 4*i + 3] = h**3   # d_i
        b[row] = f[i+1]
    
    # уравнение 3: непрерывность первых производных
    # s'_i(x_{i+1}) = s'_{i+1}(x_{i+1})
    for i in range(n-1):
        row = 2*n + i
        h_i = x[i+1] - x[i]
        h_ip1 = x[i+2] - x[i+1]
        
        a[row, 4*i + 1] = 1          # b_i
        a[row, 4*i + 2] = 2*h_i      # 2*c_i*h_i
        a[row, 4*i + 3] = 3*h_i**2   # 3*d_i*h_i^2
        a[row, 4*(i+1) + 1] = -1     # -b_{i+1}
        b[row] = 0
    
    # уравнение 4: непрерывность вторых производных
    # s''_i(x_{i+1}) = s''_{i+1}(x_{i+1})
    for i in range(n-1):
        row = 3*n - 1 + i
        h_i = x[i+1] - x[i]
        
        a[row, 4*i + 2] = 2         # 2*c_i
        a[row, 4*i + 3] = 6*h_i     # 6*d_i*h_i
        a[row, 4*(i+1) + 2] = -2    # -2*c_{i+1}
        b[row] = 0
    
    # граничные условия (естественный сплайн)
    # s''_0(x_0) = 0 и s''_{n-1}(x_n) = 0
    row1 = 4*n - 2
    a[row1, 2] = 2  # 2*c_0 = 0
    
    row2 = 4*n - 1
    h_last = x[n] - x[n-1]
    a[row2, 4*(n-1) + 2] = 2         # 2*c_{n-1}
    a[row2, 4*(n-1) + 3] = 6*h_last  # 6*d_{n-1}*h_{n-1}
    
    return a, b

def solve_spline_coefficients(x, f):
    """
    решает систему уравнений для нахождения коэффициентов сплайнов
    """
    a, b = create_spline_matrix(x, f)
    coefficients = np.linalg.solve(a, b)
    return coefficients.reshape(-1, 4)  # преобразуем в матрицу [n x 4]

# данные из условия
x = np.array([0, 1, 2, 3])
f = np.array([2, 3, 10, 29])

# создаем матрицу и решаем систему
a, b = create_spline_matrix(x, f)
coefficients = solve_spline_coefficients(x, f)

# создаем dataframe для красивого отображения матрицы
matrix_df = pd.DataFrame(a, 
                        columns=[f'coef_{i}' for i in range(12)],
                        index=[f'eq_{i}' for i in range(12)])

# выводим результаты
print("Матрица системы 12x12:")
print(matrix_df)
print("\nВектор правой части:")
print(pd.Series(b, index=[f'eq_{i}' for i in range(12)]))
print("\nКоэффициенты сплайнов (по строкам: [a_i, b_i, c_i, d_i] для каждого интервала):")
print(pd.DataFrame(coefficients, 
                  columns=['a_i', 'b_i', 'c_i', 'd_i'],
                  index=[f'interval {i}' for i in range(len(x)-1)]))

# проверка размерности
print(f"\nРазмер матрицы: {a.shape}")
print(f"Количество интервалов: {len(x)-1}")
