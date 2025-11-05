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
