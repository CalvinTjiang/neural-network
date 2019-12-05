import numpy as np
import numpy.ma as ma

class Layer:
    ''' Class to represent a layer in a neural network
    
    Static Attributes:
        SIGMOID (int): 0, Logistic (sigmoid) function
        TANH (int): 1, Hyperbolic Tangent (tanh) function
        RELU (int): 2, Rectified Linear Unit (ReLU) function
        SWISH (int): 3, Self-Gated Activation (Swish) function
        SOFTMAX (int): 4, Normalized Exponential (softmax) function

    Attributes:
        total_nodes (int): total node, have to be greater than 0
        weight (numpy.ndarray): each node's weight, 
            Shape: (self.total_nodes, previous layer's total_nodes + 1)
            Default : None

        activation (numpy.ndarray): last activation value of input data,
            Shape: (total input data, self.total_nodes + 1)
            Default : None

        z (numpy.ndarray): linear combination of each node's weight and the last input data,
            Shape: (total input data, self.total_nodes)
            Default : None

        error (numpy.ndarray): each node's error rate,
            Shape: (total input data, self.total_nodes)
            Default : None

        derivative (numpy.ndarray): each node's derivative value
            Shape: (total input data, self.total_nodes + 1)
            Default : None

        activation_type (int): type of activation function
            Default : Layer.SIGMOID (0)

    '''
    # List of activation function type
    SIGMOID = 0
    TANH = 1
    RELU = 2 
    SWISH = 3
    SOFTMAX = 4

 
    def __init__(self, total_nodes):
        ''' Class __init__ method

        Arguments:
            total_nodes (int): total node, have to be greater than 0
        '''
        self.total_nodes = total_nodes
        self.weight = None 
        self.activation = None          
        self.z = None
        self.error = None
        self.derivative = None
        self.activation_type = Layer.SIGMOID


    def activate(self, previous_layer_activation):
        ''' Calculate the layer's activation value of input data
        
        - Calculate the linear combination and stored the result in self.z
            layer's linear combination = previous layer's activation value @ layer's weight.T

        - Calculate the activation value depend on type of activation function

        Arguments:
            previous_layer_activation (numpy.ndarray): previous layer's last activation value of input data,
                Shape: (total input data, previous layer's total_nodes + 1)
        '''
        # Calculate the linear function of the layer weight and previous layer activation
        # Transpose the result, so that it have the same shape with previous layer activation
        self.z = previous_layer_activation @ self.weight.T

        # Check activation function type
        if self.activation_type == Layer.SIGMOID:
            self.activation = 1 / (1 + np.exp(-self.z))
        
        elif self.activation_type == Layer.TANH:
            self.activation = np.tanh(self.z)

        elif self.activation_type == Layer.RELU:
            self.activation = np.where(self.z > 0, self.z, 0)

        elif self.activation_type == Layer.SWISH:
            self.activation = self.z / (1 + np.exp(-self.z))
            
        elif self.activation_type == Layer.SOFTMAX:
            exponent = np.exp(self.z)
            self.activation = exponent / np.sum(exponent, axis=1)[:, np.newaxis]
        
        # Create a vector of ones
        ones = np.ones((self.activation.shape[0], 1))

        # Append the vector of ones to the activation
        self.activation = np.hstack((ones, self.activation))


    def calculate_error(self, next_layer_error, next_layer_weight):
        ''' Calculate layer's error rate 

        layer's error rate = (next layer's error rate @ next layer's weight[:, 1:]) * derivative of the activation function

        Arguments:
            next_layer_error (numpy.ndarray): previous layer's last activation value of input data,
                Shape: (total input data, next layer's total_nodes)

            next_layer_weight (numpy.ndarray): previous layer's last activation value of input data,
                Shape: (total input data, next layer's total_nodes + 1)
        '''
        # g is the derivative of the activation function
        g = None
        if self.activation_type == Layer.SIGMOID:
            g = self.activation[:, 1:] * (1 - self.activation[:, 1:])
        
        elif self.activation_type == Layer.TANH:
            g = 1 - (self.activation[:, 1:] * self.activation[:, 1:])

        elif self.activation_type == Layer.RELU:
            g = np.where(self.z > 0, self.z, 0)
            g = np.where(g <= 0, g, 1)

        elif self.activation_type == Layer.SWISH:
            g = (1 + self.z - self.activation[:, 1:]) * (self.activation[:, 1:]/ self.z)

        # Should only be used for the last layer
        elif self.activation_type == Layer.SOFTMAX:
            # Should be used for the last layer
            pass

        else:
            raise ValueError(f'{self.activation_type} is not a valid activation_type')

        # Calculate the error
        self.error = (next_layer_error @ next_layer_weight[:, 1:]) * g


    def derive(self, previous_layer_activation, total_data, learning_rate):
        ''' Calculate each node's derivative
        
        layer's derivative = (layer's error.T @ previous layer's activation value) / total_data
        layer's weight = layer's weight - learning_rate * layer's derivative

        Arguments:
            previous_layer_activation (numpy.ndarray): previous layer's last activation value of input data,
                Shape: (total input data, next layer's total_nodes)

            total_data (int): total_data, have to be greater than 0
            learning_rate (float): layer's learning rate, have to be greater or equal to zero
        '''
        self.derivative = (self.error.T @ previous_layer_activation) / total_data
        self.weight = self.weight - learning_rate * self.derivative


class NeuralNetwork:
    ''' Class to represent a layer in a neural network

    Attributes:
        layers (list): list of all Layer object
        learning_rate (float): learning rate
            Default: 1e-3 (0.001)
            
        batch_size (int): size of data for each training
            Default: 100

        total_epochs (int): total number of entire dataset are being used for training
            Default: 1
    '''

    def __init__(self):
        self.layers = []
        self.learning_rate = 1e-3
        self.batch_size = 100
        self.total_epochs = 1


    def forward_propagation(self, x):
        ''' Apply forward propagation to calculate each layer's activation value

        Arguments:
            x (numpy.ndarray): input data,
                Shape: (total input data, input layer's total_nodes + 1)
        
        Precondition:
            Input layer and output layer.
        '''
        if len(self.layers) <= 1:
            raise Exception("Need at least input layer and output layer")

        # Create a vector of ones
        ones = np.ones((x.shape[0], 1))

        # Append the vector of ones to the activation
        self.layers[0].activation = np.hstack((ones, x))
        for i in range(1, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1].activation)


    def back_propagation(self, y):
        ''' Apply back propagation to calculate the error rate

        Arguments:
            y (numpy.ndarray): the correct output data,
                Shape: (total input data, output layer's total_nodes)
        
        Precondition:
            Apply forward propagation with the same dataset.
            Input layer and output layer.
        '''
        if len(self.layers) <= 1:
            raise Exception("Need at least input layer and output layer")

        self.layers[-1].error = self.layers[-1].activation[:, 1:] - y
        for i in range(len(self.layers) - 2, 0, -1):
            self.layers[i].calculate_error(self.layers[i + 1].error, self.layers[i + 1].weight)


    def partial_derivative(self, total_data):
        ''' Apply partial derivative to decrease each layer's error rate

        Arguments:
            total_data (int): total input data
        
        Precondition:
            Apply back propagation with the same dataset.
            Input layer and output layer.
        '''
        if len(self.layers) <= 1:
            raise Exception("Need at least input layer and output layer")

        for i in range(1, len(self.layers)):
            self.layers[i].derive(self.layers[i - 1].activation, total_data, self.learning_rate)


    def train(self, x, y):
        ''' Train the neural network to fit the given data

        Arguments:
            x (numpy.ndarray): input data,
                Shape: (total input data, input layer's total_nodes + 1)

            y (numpy.ndarray): the correct output data,
                Shape: (total input data, output layer's total_nodes)
        
        Precondition:
            Input layer and output layer.
        '''
        if self.batch_size <= 0:
            raise ValueError("batch_size have to be greater than 0")
        
        if len(self.layers) <= 1:
            raise Exception("Need at least input layer and output layer")

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


    def predict(self, x) -> np.ndarray:
        ''' Calculate the output layer prediction for input data

        Arguments:
            x (numpy.ndarray): input data,
                Shape: (total input data, input layer's total_nodes + 1)
        
        Precondition:
            Input layer and output layer.
        '''
        self.forward_propagation(x)
        return self.layers[-1].activation[:, 1:]


    def cost(self, x, y) -> np.ndarray:
        ''' Calculate the cost of output layer compare with correct data

        Arguments:
            x (numpy.ndarray): input data,
                Shape: (total input data, input layer's total_nodes + 1)

            y (numpy.ndarray): the correct output data,
                Shape: (total input data, output layer's total_nodes)
        
        Precondition:
            Input layer and output layer.
        '''
        prediction = self.predict(x)

        # Calculate Difference for y[i] = 1
        a = np.multiply(y, ma.log(prediction).filled(0))

        # Calculate Difference for y[i] = 0
        b = np.multiply(1 - y, ma.log(1 - prediction).filled(0))

        difference = a + b

        total_cost = -np.sum(difference, axis=0) / len(y)
        return np.sum(total_cost)


    def add_layer(self, total_nodes, activation_type, epsilon=1.0):
        ''' Append a new layer object to this neural network

        Arguments:
            total_nodes (int): new layer's total nodes

            activation_type (int): new layer's activation function type

            epsilon (float): the lower and upper bound for new layer's weight
        '''
        if total_nodes <= 0:
            raise ValueError("total_nodes have to be greater than 0")

        if epsilon <= 0:
            raise ValueError("epsilon have to be greater than 0")

        new_layer = Layer(total_nodes)
        if len(self.layers) > 0:        
            new_layer.weight = np.random.uniform(-epsilon, epsilon, (total_nodes, self.layers[-1].total_nodes + 1))
            new_layer.activation_type = activation_type
        self.layers.append(new_layer)
