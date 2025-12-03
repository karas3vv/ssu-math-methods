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
