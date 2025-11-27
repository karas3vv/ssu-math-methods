V = 5

def derivative(x):
    return -(4 * V * x**4 - 3 * V**2 * x**3 + 6 * V * x - 2 * V**2)

def Y(x):
    return V * x**2 * (x - V)

def p(x):
    return -x**2

def q(x):
    return -x

def main():
    n = 1000
    x0 = 0
    h = V / n
    x = [x0 + i * h for i in range(n + 1)]
    exact = [Y(xi) for xi in x]

    f = [0.0] * (n + 1)
    s = [0.0] * (n + 1)
    t = [0.0] * (n + 1)
    r = [0.0] * (n + 1)
    f1 = [0.0] * (n + 1)
    s1 = [0.0] * (n + 1)
    y = [0.0] * (n + 1)
    e = [0.0] * (n + 1)

    for i in range(1, n):
        f[i] = 0.5 * (1 + 0.5 * h * p(x[i]))
        s[i] = 0.5 * (1 - 0.5 * h * p(x[i]))
        t[i] = 1 + 0.5 * h**2 * q(x[i])
        r[i] = 0.5 * h**2 * derivative(x[i])

    f1[1] = 0.0
    s1[1] = 0.0

    for j in range(1, n):
        denom = t[j] - f[j] * f1[j]
        f1[j + 1] = s[j] / denom
        s1[j + 1] = (r[j] + f[j] * s1[j]) / denom

    # Граничное условие y[n] = 0 в оригинале неявно, принимаем y[n]=0
    y[n] = 0.0
    for j in range(n - 1, 0, -1):
        y[j] = f1[j + 1] * y[j + 1] + s1[j + 1]

    max_e = 0.0
    max_e_index = 0
    for i in range(n + 1):
        e[i] = abs(y[i] - exact[i])
        if (e[i] > max_e):
            max_e = e[i]
            max_e_index = i

    print("x\t y\t\t exact\t e")
    for i in range(n + 1):
        print(f"{x[i]:.2f}\t {y[i]:.2f} \t{exact[i]:.2f}\t {e[i]:.8f}")
    print("Максимальный e: ", max_e, "Номер максимального е: ", max_e_index)

if __name__ == "__main__":
    main()
