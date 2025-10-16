import numpy as np
import pandas as pd

# Входные данные
x_data = np.array([0, 1, 2, 3])
f_data = np.array([2, 3, 10, 29])

print("Исходные данные:")
print(pd.DataFrame({'X': x_data, 'F': f_data}))

# -------------------------------
# Построение таблицы разделённых разностей
# -------------------------------
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

# -------------------------------
# Коэффициенты Ньютона (верхняя строка таблицы)
# -------------------------------
newton_coeffs = dd_table[0,:]
print("\nКоэффициенты многочлена Ньютона:")
print(newton_coeffs)

# -------------------------------
# Функция для вычисления значения многочлена Ньютона
# -------------------------------
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
