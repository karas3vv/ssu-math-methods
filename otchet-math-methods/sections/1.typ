*Задание 1.* По данным интерполяции построить интерполяционный многочлен в общем виде.
```python
import numpy

# входные данные
x_data = [0, 1, 2, 3]
f_data = [2, 3, 10, 29]
x_np_data = numpy.array(x_data)
f_np_data = numpy.array(f_data)

print("Исходные данные:")
for i in range (0, 4):
    print('X' + str(i) + ': ' + str(x_data[i]) + '\tF' + str(i) + ': ' + str(f_data[i]))

# сторим матрицу
matrix = numpy.vander(x_np_data, len(x_data), increasing=True)
matrix_flipped = numpy.flip(matrix, axis=1)
print("\nСистема построена:")
for i in range(0, 4):
    s = ""
    for j in range (0, 4):
        s += str(matrix_flipped[i][j]) + ' * a' + str(4 - j - 1) + ' + '
    print(s[:-2] + ' = ' + str(f_data[i]))

print()

# решение системы уравнений для нахождения коэффициентов многочлена
coefficients = numpy.linalg.solve(matrix, f_np_data)
print("\nКоэффициенты найдены:")
for i in range (0, 4):
    print('a' + str(i) + ': ' + str(coefficients[i]))

print("\nP3 (x): ")
for i in range (0, 7):
    print('x = ' + str(i / 2) + ': ' + str(
        coefficients[3] * (i / 2) * (i / 2) * (i / 2)
        + coefficients[2] * (i / 2) * (i / 2)
        + coefficients[1] * (i / 2)
        + coefficients[0]
        ))
```

```
Исходные данные:
X0: 0   F0: 2
X1: 1   F1: 3
X2: 2   F2: 10
X3: 3   F3: 29

Система построена:
0 * a3 + 0 * a2 + 0 * a1 + 1 * a0  = 2
1 * a3 + 1 * a2 + 1 * a1 + 1 * a0  = 3
8 * a3 + 4 * a2 + 2 * a1 + 1 * a0  = 10
27 * a3 + 9 * a2 + 3 * a1 + 1 * a0  = 29


Коэффициенты найдены:
a0: 2.0
a1: 0.0
a2: -0.0
a3: 1.0

P3 (x): 
x = 0.0: 2.0
x = 0.5: 2.125
x = 1.0: 3.0
x = 1.5: 5.375
x = 2.0: 10.0
x = 2.5: 17.625
x = 3.0: 29.0
```

*Задание 2.* По данным интерполяции с предыдущего задания построить интерполяционный многочлен в форме Лагранжа.
```python
import numpy

def lagrange_fundamental(k, x_nodes, z):
        Lk = 1.0
        for i, xi in enumerate(x_nodes):
            if i != k:
                Lk *= (z - xi) / (x_nodes[k] - xi)
        return Lk

def lagrange_interpolation(x_nodes, y_nodes, z):
        P = 0.0
        for k in range(len(x_nodes)):
            P += y_nodes[k] * lagrange_fundamental(k, x_nodes, z)
        return P

x_data = [0, 1, 2, 3]
f_data = [2, 3, 10, 29]
x_np_data = numpy.array(x_data)
f_np_data = numpy.array(f_data)

print("Входные данные:")
for i in range (0, 4):
    print('X' + str(i) + ': ' + str(x_data[i]) + '\tF' + str(i) + ': ' + str(f_data[i]))
    
print("\nL3 (x): ")
for i in range (0, 7):
    print('x = ' + str(i / 2) + ': ' + str(
        lagrange_interpolation(x_data, f_data, i / 2)
        ))
```

```
Входные данные:
X0: 0   F0: 2
X1: 1   F1: 3
X2: 2   F2: 10
X3: 3   F3: 29

L3 (x): 
x = 0.0: 2.0
x = 0.5: 2.125
x = 1.0: 3.0
x = 1.5: 5.375
x = 2.0: 10.0
x = 2.5: 17.625
x = 3.0: 29.0
```

*Задание 3.* По данным интерполяции из предыдущих двух заданий построить интерполяционный многочлен в форме Ньютона.
```python 
import numpy as np
import pandas as pd

# Входные данные
x_data = np.array([0, 1, 2, 3])
f_data = np.array([2, 3, 10, 29])

print("Исходные данные:")
print(pd.DataFrame({'X': x_data, 'F': f_data}))

# Построение таблицы разделённых разностей
def divided_difference_table(x, y):
    n = len(x)
    table = np.zeros((n, n))
    table[:,0] = y
    for j in range(1, n):
        for i in range(n - j):
            table[i,j] = (table[i+1,j-1] - table[i,j-1]) / (x[i+j] - x[i])
    return table

dd_table = divided_difference_table(x_data, f_data)

# Вывод таблицы разделённых разностей
dd_df = pd.DataFrame(dd_table, columns=[f"f[x0..x{j}]" for j in range(len(x_data))])
print("\nТаблица разделённых разностей (матрица Ньютона):")
print(dd_df)

# Коэффициенты Ньютона (верхняя строка таблицы)
newton_coeffs = dd_table[0,:]
print("\nКоэффициенты многочлена Ньютона:")
print(newton_coeffs)

# Функция для вычисления значения многочлена Ньютона
def newton_polynomial(x, coeffs, x_data):
    n = len(coeffs) - 1
    p = coeffs[n]
    for k in range(1, n + 1):
        p = coeffs[n-k] + (x - x_data[n-k]) * p
    return p

# Вычисление значений на промежуточных точках
x_half = [(x_data[i] + x_data[i+1])/2 for i in range(len(x_data)-1)]
f_half_newton = [newton_polynomial(x, newton_coeffs, x_data) for x in x_half]

# Объединяем узлы и промежуточные точки
x_combined = np.concatenate([x_data, x_half])
y_newton = np.concatenate([f_data, f_half_newton])

# Сортируем по X
sort_idx = np.argsort(x_combined)
x_combined = x_combined[sort_idx]
y_newton = y_newton[sort_idx]

# Создаем итоговую таблицу
newton_df = pd.DataFrame({
    'X': x_combined,
    'F (Ньютон)': y_newton
})

print("\nЗначения многочлена Ньютона:")
print(newton_df.to_string(index=False))
```

```
Исходные данные:
   X   F
0  0   2
1  1   3
2  2  10
3  3  29

Таблица разделённых разностей (матрица Ньютона):
   f[x0..x0]  f[x0..x1]  f[x0..x2]  f[x0..x3]
0        2.0        1.0        3.0        1.0
1        3.0        7.0        6.0        0.0
2       10.0       19.0        0.0        0.0
3       29.0        0.0        0.0        0.0

Коэффициенты многочлена Ньютона:
[2. 1. 3. 1.]

Значения многочлена Ньютона:
  X  F (Ньютон)
0.0       2.000
0.5       2.125
1.0       3.000
1.5       5.375
2.0      10.000
2.5      17.625
3.0      29.000
```

*Задание 4.* По данным интерполяции с предыдущего задания построить кусочно-непрерывную склейку кубических сплайнов.
```python

import numpy as np
import pandas as pd

def create_spline_matrix(x, f):
    n = len(x) - 1  # количество интервалов
    
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
```

```
Матрица системы 12x12:
       coef_0  coef_1  coef_2  coef_3  coef_4  coef_5  coef_6  coef_7  coef_8  coef_9  coef_10  coef_11
eq_0      1.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0      0.0      0.0
eq_1      0.0     0.0     0.0     0.0     1.0     0.0     0.0     0.0     0.0     0.0      0.0      0.0
eq_2      0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     1.0     0.0      0.0      0.0
eq_3      1.0     1.0     1.0     1.0     0.0     0.0     0.0     0.0     0.0     0.0      0.0      0.0
eq_4      0.0     0.0     0.0     0.0     1.0     1.0     1.0     1.0     0.0     0.0      0.0      0.0
eq_5      0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     1.0     1.0      1.0      1.0
eq_6      0.0     1.0     2.0     3.0     0.0    -1.0     0.0     0.0     0.0     0.0      0.0      0.0
eq_7      0.0     0.0     0.0     0.0     0.0     1.0     2.0     3.0     0.0    -1.0      0.0      0.0
eq_8      0.0     0.0     2.0     6.0     0.0     0.0    -2.0     0.0     0.0     0.0      0.0      0.0
eq_9      0.0     0.0     0.0     0.0     0.0     0.0     2.0     6.0     0.0     0.0     -2.0      0.0
eq_10     0.0     0.0     2.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0      0.0      0.0
eq_11     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0      2.0      6.0

Вектор правой части:
eq_0      2.0
eq_1      3.0
eq_2     10.0
eq_3      3.0
eq_4     10.0
eq_5     29.0
eq_6      0.0
eq_7      0.0
eq_8      0.0
eq_9      0.0
eq_10     0.0
eq_11     0.0

Коэффициенты сплайнов (по строкам: [a_i, b_i, c_i, d_i] для каждого интервала):
             a_i   b_i  c_i  d_i
interval 0   2.0   0.2  0.0  0.8
interval 1   3.0   2.6  2.4  2.0
interval 2  10.0  13.4  8.4 -2.8

Размер матрицы: (12, 12)
Количество интервалов: 3
```
*Задание 5.* Решить СЛАУ методом Гаусса. 
```python
import numpy as np

A = np.array([[5, 0.05, 0.05, 0.05, 0.05],
             [0.06, 6, 0.06, 0.06, 0.06],
             [0.07, 0.07, 7, 0.07, 0.07],
             [0.08, 0.08, 0.08, 8, 0.08,],
             [0.09, 0.09, 0.09, 0.09, 9]])
print("1) Матрица A:\n", A)
print("Определитель матрицы A = ", np.linalg.det(A))
b = np.zeros(len(A))
print("\nКолонка b:")
for i in range(len(A)):
    b[i] = A[i][i]
    print("[", b[i], "]")
b = np.dot(A, b)

def forward_elimination(A, b):
    n = len(b)
    for k in range(n - 1):
        for i in range(k + 1, n):
            if A[i, k] != 0:
                factor = A[i, k] / A[k, k]
                A[i, k+1:n] = A[i, k+1:n] - factor * A[k, k+1:n]
                A[i, k] = 0  # Explicitly zero out for clarity
                b[i] = b[i] - factor * b[k]
    
    return A, b

A_tmp, b_tmp = forward_elimination(A, b)
print("Матрица после прямого прохода:")
print(A_tmp)

def backward_substitution(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] /= U[i, i]
    return x

solution = backward_substitution(A_tmp, b_tmp)

print("Решение (A|b) методом Гаусса")
print("Вектор решения после обратного прохода: ")

for i in range(len(solution)):
    print("x", str(i), "=", solution[i])

print(solution, "\n 4) x =")
for i in range(len(solution)):
    print("[", solution[i], "]")
b = np.dot(A, b)
```

```
1) Матрица A:
 [[5.   0.05 0.05 0.05 0.05]
 [0.06 6.   0.06 0.06 0.06]
 [0.07 0.07 7.   0.07 0.07]
 [0.08 0.08 0.08 8.   0.08]
 [0.09 0.09 0.09 0.09 9.  ]]
Определитель матрицы A =  15105.180138048006

Колонка b:
[ 5.0 ]
[ 6.0 ]
[ 7.0 ]
[ 8.0 ]
[ 9.0 ]
Матрица после прямого прохода:
[[5.         0.05       0.05       0.05       0.05      ]
 [0.         5.9994     0.0594     0.0594     0.0594    ]
 [0.         0.         6.99861386 0.06861386 0.06861386]
 [0.         0.         0.         7.99764706 0.07764706]
 [0.         0.         0.         0.         8.99650485]]
Решение (A|b) методом Гаусса
Вектор решения после обратного прохода: 
x 0 = 5.0
x 1 = 5.999999999999999
x 2 = 7.000000000000001
x 3 = 7.999999999999999
x 4 = 9.0
[5. 6. 7. 8. 9.] 
 4) x =
[ 5.0 ]
[ 5.999999999999999 ]
[ 7.000000000000001 ]
[ 7.999999999999999 ]
[ 9.0 ]
```

*Задание 6.* Решить СЛАУ с помощью прогона.
```python
import numpy as np

# Исходные данные
A = np.array([[5, 0.05, 0, 0, 0],
              [0.06, 6, 0.06, 0, 0.],
              [0, 0.07, 7, 0.07, 0],
              [0, 0, 0.08, 8, 0.08],
              [0, 0, 0, 0.09, 9]])

# Вычисляем вектор b
b = np.zeros(len(A))
for i in range(len(A)):
    b[i] = A[i][i]
b = np.dot(A, b.reshape(-1,1))

print("Матрица A:")
print(A)
print("Столбец b:")
print(b)

Q = np.zeros(len(A))
P = np.zeros(len(A) - 1)
P[0] = -A[0][1]/A[0][0]
Q[0] = b[0][0]/A[0][0]

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
```

```
Матрица A:
[[5.   0.05 0.   0.   0.  ]
 [0.06 6.   0.06 0.   0.  ]
 [0.   0.07 7.   0.07 0.  ]
 [0.   0.   0.08 8.   0.08]
 [0.   0.   0.   0.09 9.  ]]
Столбец b:
[[25.3 ]
 [36.72]
 [49.98]
 [65.28]
 [81.72]]
Прямая прогонка
Список Pi и Qi
[-0.01 -0.010001 -0.010001 -0.010001] [5.06 6.070007 7.080008 8.090009 9.]
Обратная прогонка
x 5 = 9.0
x 4 =  8.000000000000002
x 3 =  6.999999999999999
x 2 =  6.0
x 1 =  5.000000000000001
[[5.]
 [6.]
 [7.]
 [8.]
 [9.]]
```

*Задание 7.* Решить СЛАУ методом простой итерации.