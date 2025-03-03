
README

Description
This project demonstrates the implementation of a simple deep neural network for binary classification using custom activation functions, loss functions, and backpropagation. The code includes data generation, parameter initialization, forward propagation, and gradient descent training.

Key Features:
- Generates a dataset using Gaussian quantiles.
- Implements activation functions (ReLU and Sigmoid) and Mean Squared Error (MSE) as the loss function.
- Uses backpropagation for training with gradient descent.

Dependencies
The following Python libraries are required:
- numpy: For matrix operations and handling numerical data.
- matplotlib: For plotting the dataset (optional, not used in training).
- sklearn.datasets: To generate synthetic datasets (Gaussian quantiles).

Install the required libraries with:
pip install numpy matplotlib scikit-learn

Code Structure
1. generate_data(N=1000):
   - Generates a dataset with N samples using Gaussian quantiles.
   - Returns X (features) and Y (labels).

2. Activation Functions:
   - sigmoid(x, derivate=False): Implements the Sigmoid activation function and its derivative.
   - relu(x, derivate=False): Implements the ReLU activation function and its derivative.

3. Loss Function:
   - mse(y, y_hat, derivate=False): Calculates Mean Squared Error and its derivative for backpropagation.

4. initialize_parameters_deep(layers_dims):
   - Initializes the weights and biases for the neural network based on the specified layer dimensions (layers_dims).

5. train(x_data, y_data, learning_rate, params, training=True):
   - Performs forward propagation, backpropagation, and updates the weights and biases using gradient descent.

Usage
1. Generate Data:
   X, Y = generate_data(N=1000)

2. Initialize Parameters:
   Define the dimensions of each layer (e.g., input layer, hidden layers, and output layer).
   layers_dims = [2, 5, 3, 1]  # Example: input -> 2, hidden layers -> 5 and 3, output -> 1
   parameters = initialize_parameters_deep(layers_dims)

3. Training the Model:
   Train the model by calling the train function:
   output = train(X, Y, learning_rate=0.01, params=parameters, training=True)

4. Prediction:
   Use the trained model to make predictions by running the train function with training=False:
   predictions = train(X, Y, learning_rate=0.01, params=parameters, training=False)

Customization
- Network Structure: Modify the layers_dims variable to change the number of layers and neurons.
- Learning Rate: Adjust the learning_rate to control the speed of convergence during training.
- Activation Functions: You can modify or add new activation functions in the code if needed.

License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.
