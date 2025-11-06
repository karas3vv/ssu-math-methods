import numpy as np

A = np.array([[5, 0.05, 0.05, 0.05, 0.05],
             [0.06, 6, 0.06, 0.06, 0.06],
             [0.07, 0.07, 7, 0.07, 0.07],
             [0.08, 0.08, 0.08, 8, 0.08,],
             [0.09, 0.09, 0.09, 0.09, 9]])

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

alpha = A.copy()
beta = b.copy()
for i in range(len(A)):
    for j in range(len(A)):
        if i == j:
            alpha[i][j] = 0
        else:
            alpha[i][j] = - A[i][j]/A[i][i]
print("Матрица alpha:")
print(alpha)

for i in range(len(A)):
    beta[i][0] = b[i][0]/A[i][i]
print("Столбец beta:")
print(beta)
epsilon = 10**-9
print("Считаем до точности epsilon=", epsilon)
xk = np.zeros(len(A)).reshape(-1,1)
# xk = np.copy(b)

print("x^(0) = ", xk)
def normStop (xk, xkp1, epsilon):
    return max(np.abs(np.add(xk, -xkp1))) < epsilon
for i in range(17):
    xkp1 = np.add(np.dot(alpha, xk), beta) 
    print(f"x^({i+1})=", xkp1)
    if normStop(xk, xkp1, epsilon):
        break
    xk = xkp1
