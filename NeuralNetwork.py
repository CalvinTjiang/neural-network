import numpy as np
import numpy.ma as ma
class Layer:
    # List of activation function type
    SIGMOID = 0
    TANH = 1
    RELU = 2 
    SOFTMAX = 3

 
    def __init__(self, total_nodes):
        self.total_nodes = total_nodes
        self.weight = None
        self.activation = None
        self.z = None
        self.error = None
        self.derivative = None
        self.activation_type = Layer.SIGMOID


    def activate(self, previous_layer_activation):
        # Calculate the linear function of the layer weight and previous layer activation
        # Transpose the result, so that it have the same shape with previous layer activation
        self.z = previous_layer_activation @ self.weight.T

        # Check activation function type
        if self.activation_type == Layer.SIGMOID:
            self.activation = (1 / (1 + np.exp(-self.z)))
        
        elif self.activation_type == Layer.TANH:
            self.activation = np.tanh(self.z)

        elif self.activation_type == Layer.RELU:
            self.activation = np.where(self.z > 0, self.z, 0)
            
        elif self.activation_type == Layer.SOFTMAX:
            exponent = np.exp(self.z)
            self.activation = exponent / np.sum(exponent, axis=1)[:, np.newaxis]
        
        # Create a vector of ones
        ones = np.ones((self.activation.shape[0], 1))

        # Append the vector of ones to the activation
        self.activation = np.hstack((ones, self.activation))


    def calculate_error(self, next_layer_error, next_layer_weight):
        # g is the derivative of the activation function
        g = None
        if self.activation_type == Layer.SIGMOID:
            g = self.activation[:, 1:] * (1 - self.activation[:, 1:])
        
        elif self.activation_type == Layer.TANH:
            g = 1 - (self.activation[:, 1:] * self.activation[:, 1:])

        elif self.activation_type == Layer.RELU:
            g = np.where(self.z > 0, self.z, 0)
            g = np.where(g <= 0, g, 1)
            
        elif self.activation_type == Layer.SOFTMAX:
            # Should be used for the last layer
            pass

        # Calculate the error
        next_layer_weight = next_layer_weight[:, 1:]

        self.error = (next_layer_error @ next_layer_weight) * g


    def derive(self, previous_layer_activation, total_data, learning_rate):
        self.derivative = (self.error.T @ previous_layer_activation) / total_data
        self.weight = self.weight - learning_rate * self.derivative


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.learning_rate = 1e-3
        self.batch_size = 0
        self.total_epochs = 0


    def forward_propagation(self, x):
        # Create a vector of ones
        ones = np.ones((x.shape[0], 1))

        # Append the vector of ones to the activation
        self.layers[0].activation = np.hstack((ones, x))
        for i in range(1, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1].activation)


    def back_propagation(self, y):
        self.layers[-1].error = self.layers[-1].activation[:, 1:] - y
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].calculate_error(self.layers[i + 1].error, self.layers[i + 1].weight)


    def partial_derivative(self, total_data):
        for i in range(1, len(self.layers)):
            self.layers[i].derive(self.layers[i - 1].activation, total_data, self.learning_rate)


    def train(self, x, y):
        if self.batch_size <= 0:
            self.batch_size = len(x)
        for epoch in range(self.total_epochs):
            # initial start batch is 0
            start_batch = 0 
            end_batch = self.batch_size
            while start_batch < len(x):
                # if the end_batch is more than the length of x 
                if end_batch >= len(x):
                    end_batch = len(x)

                # Divide the train data into smaller batch
                current_x = x[start_batch:end_batch]
                current_y = y[start_batch:end_batch]
                
                # Train the network with the batch data
                self.forward_propagation(current_x)
                self.back_propagation(current_y)
                self.partial_derivative(len(current_y))

                # Move to next batch
                start_batch += self.batch_size
                end_batch += self.batch_size


    def predict(self, x):
        self.forward_propagation(x)
        return self.layers[-1].activation[:, 1:]


    def cost(self, x, y):
        prediction = self.predict(x)

        # Calculate Difference for y[i] = 1
        a = np.multiply(y, ma.log(prediction).filled(0))

        # Calculate Difference for y[i] = 0
        b = np.multiply(1 - y, ma.log(1 - prediction).filled(0))

        difference = a + b

        total_cost = -np.sum(difference, axis=0) / len(y)
        return np.sum(total_cost)


    def add_layer(self, total_nodes, activation_type, epsilon=1):
        new_layer = Layer(total_nodes)
        if len(self.layers) > 0:        
            new_layer.weight = np.random.uniform(-epsilon, epsilon, (total_nodes, self.layers[-1].total_nodes + 1))
            new_layer.activation_type = activation_type
        self.layers.append(new_layer)
