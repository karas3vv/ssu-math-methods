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