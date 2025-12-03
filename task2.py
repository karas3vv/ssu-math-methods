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
