import matplotlib.pyplot as plt
from src.Neural_Networks_Numpy import generate_data, initialize_parameters_deep, train, mse

# Generar los datos
X, Y = generate_data(N=1000)

# Inicializar parámetros de la red
layers_dims = [2, 6, 10, 1]
params = initialize_parameters_deep(layers_dims)
error = []

# Entrenar la red
for i in range(50000):
    output = train(X, Y, 0.001, params)
    if i % 50 == 0:
        loss = mse(Y, output)
        print(f"Iteración {i}: Error: {loss}")
        error.append(loss)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()

# Graficar el error
plt.plot(error)
plt.xlabel("Iteraciones")
plt.ylabel("Error")
plt.show()
