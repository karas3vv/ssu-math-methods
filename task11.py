import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.linalg import solve


def solve_fredholm_degenerate_correct(V):
    a, b = 0, 1
    n = 3
    lam = 1

    def y_exact(x):
        return V * x

    def f(x):
        return V * (4 / 3 * x + 1 / 4 * x**2 + 1 / 5 * x**3)

    a_funcs = [lambda x: x, lambda x: x**2, lambda x: x**3]
    b_funcs = [lambda t: t, lambda t: t**2, lambda t: t**3]
    alpha = np.zeros((n, n))
    gamma = np.zeros(n)
    for i in range(n):
        for k in range(n):
            alpha[i, k], _ = quad(lambda x: a_funcs[i](x) * b_funcs[k](x), a, b)
        gamma[i], _ = quad(lambda x: f(x) * b_funcs[i](x), a, b)
    A = np.eye(n) + lam * alpha.T
    q = solve(A, gamma)

    def y_numerical(x):
        result = f(x)
        for i in range(n):
            result -= lam * q[i] * a_funcs[i](x)
        return result

    x_test = np.linspace(a, b, 10)
    y_num_vals = [y_numerical(x) for x in x_test]
    y_ex_vals = [y_exact(x) for x in x_test]
    errors = [abs(y_num_vals[i] - y_ex_vals[i]) for i in range(len(x_test))]
    df = pd.DataFrame(
        {"x": x_test, "y_метода": y_num_vals, "y_точн": y_ex_vals, "eps": errors}
    )
    print(df.to_string(index=False))
    print("\n")
    return y_numerical, y_exact

if __name__ == "__main__":
    V = 5  # номер варанта
    solve_fredholm_degenerate_correct(V)
