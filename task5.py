import numpy as np

A = np.array([[5, 0.5, 0.5, 0.5, 0.5],
             [0.6, 6, 0.6, 0.6, 0.6],
             [0.7, 0.7, 7, 0.7, 0.7],
             [0.8, 0.8, 0.8, 8, 0.8],
             [0.9, 0.9, 0.9, 0.9, 9]])
print(A)
print("Определитель матрицы A")
print(np.linalg.det(A))
b = np.zeros(len(A))
for i in range(len(A)):
    b[i] = A[i][i]
b = np.dot(A, b)
print("Столбец b:")
print(b)

def forward_elimination(A, b):
    n = len(b)
    for k in range(n - 1):
        for i in range(k + 1, n):
            if A[i, k] != 0:
                factor = A[i, k] / A[k, k]
                A[i, k+1:n] = A[i, k+1:n] - factor * A[k, k+1:n]
                A[i, k] = 0
                b[i] = b[i] - factor * b[k]
    print("Матрица после прямого прохода:")
    print(A)
    return A, b

A_tmp, b_tmp = forward_elimination(A, b)

def backward_substitution(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = b[i] - np.dot(U[i, i+1:], x[i+1:])
        x[i] /= U[i, i]
    return x

solution = backward_substitution(A_tmp, b_tmp)

print("Вектор решения после обратного прохода:")

for i in range(len(solution)):
    print("x", str(i), "=", solution[i])

print("Решение (A|b) методом Гаусса")
print(np.linalg.solve(A,b))
