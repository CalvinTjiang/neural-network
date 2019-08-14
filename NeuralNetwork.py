import numpy as np

class Layer:
    def __init__(self, weight):
        self.total_node = len(weight)
        self.weight = weight
        self.activation = np.ones(self.total_node + 1)
        self.error = np.ones(self.total_node + 1)
        self.derivate = np.zeros(weight.shape)


    def activate(self, x):
        '''
        Activate the layer using x as input

        Parameter :
            x : input data -> Numpy Array
        '''
        self.activation[1:] = 1 / (1 + np.exp(-self.weight @ x))


    def gradient_descent(self, x, learning_rate, regularization_rate):
        '''
        Apply gradient descent

        Parameter :
            x : input data -> Numpy Array
            learning_rate : -> Float
            reqularization_rata : -> Float 
        
        Precondition :
            Partial derivative has been applied with correct input data
        '''
        self.weight[...,0] -= (learning_rate * self.derivate[...,0] / len(x))
        self.weight[...,1:] -= (learning_rate * (self.derivate[...,1:] / len(x) + regularization_rate * self.weight[...,1:]))
        self.derivate = np.zeros(self.weight.shape)
    

    def save(self, filename):
        '''
        Save the weight of each layer to a .npy extension file

        Parameter :
            foldername : foldername -> String
        '''
        np.save(filename, self.weight)


    def load(self, filename):
        '''
        load the weight of each layer from a foldername with .npy extension

        Parameter :
            foldername : foldername -> String
        '''
        try:
            temp = np.load(filename)
            if len(temp) == len(self.weight):
                self.weight = temp
            else:
                print(f'Error : different size of weight')
        except:
            print(f'Error : {filename} not found!')



class Network:
    def __init__(self, input_node, learning_rate, regularization_rate, batch_size):
        self.layers = [Layer(np.ones([input_node, 1]))]
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate
        self.cost = 0
        self.batch_size = batch_size


    def add_layer(self, node=2, epsilon=1):
        '''
        Add new layer to the back of the neural network model

        Parameter :
            node : amount of node -> Int (default = 2)
            epsilon : range of the random value -> Int (default = 1)
        '''
        weight = np.random.rand(node, self.layers[-1].total_node + 1) 
        weight -= np.random.rand(node, self.layers[-1].total_node + 1)
        self.layers.append(Layer(weight * epsilon))


    def calculate_cost(self, x, y):
        '''
        Calculate the cost of predict result (by using input data) and input result

        Parameter :
            x : input data -> Numpy Array
            y : input result -> Numpy Array
        '''
        # There is some error here
        prediction = np.array([1])
        for data in x:
            prediction = np.vstack((prediction, self.predict(data)))
        prediction = prediction[1:]
        self.cost = (y * np.log(prediction) + (1 - y) * np.log(1 - prediction))
        self.cost = -self.cost.sum() / len(y)


    def forward_propagation(self, x):
        '''
        Apply forward propagation using x as input

        Parameter :
            x : input data -> Numpy Array
        '''
        self.layers[1].activate(x)
        for i in range(2, len(self.layers)):
            self.layers[i].activate(self.layers[i - 1].activation)
    

    def back_propagation(self, y):
        '''
        Apply back propagation by using y as the result to calculate each layer error rate

        Parameter :
            y : input result -> Numpy Array
        
        Precondition :
            forward propagation has been applied with the correct input data
        '''
        self.layers[-1].error[1:] = self.layers[-1].activation[1:] - y
        for i in range(1, len(self.layers) - 1)[::-1]:
            self.layers[i].error = self.layers[i + 1].weight[...,:].T @ self.layers[i + 1].error[1:]
            self.layers[i].error[1:] *= self.layers[i].activation[1:] * (1 - self.layers[i].activation[1:])


    def partial_derivative(self, x):
        '''
        Calculate the partial derivative of each weight in every layers

        Parameter :
            x : input data -> Numpy Array

        Precondition :
            Back propagation has been applied with correct input result
        '''
        self.layers[1].derivate += self.layers[1].error[1:,np.newaxis] * x
        for i in range(2, len(self.layers)):
            self.layers[i].derivate += self.layers[i].error[1:,np.newaxis] * self.layers[i - 1].activation


    def gradient_descent(self, x):
        '''
        Apply gradient descent to all the layers

        Parameter :
            x : input data -> Numpy Array
        
        Precondition :
            Partial derivative has been applied with correct input data
        '''
        for i in range(1, len(self.layers)):
            self.layers[i].gradient_descent(x, self.learning_rate, self.regularization_rate)
          

    def train(self, x, y, epochs=1):
        '''
        Train the network model

        Parameter :
            x : training data input -> Numpy Array
            y : training data result -> Numpy Array
            epochs : number of all data being train -> Int (default = 1)
        '''
        print('Total Data   :', len(x))
        print('Total Epochs :', epochs)
        for epoch in range(epochs):

            print('-------------------')
            print('Epoch :', epoch)
            start_batch = 0 
            end_batch = self.batch_size
            while start_batch < len(x):

                if start_batch + self.batch_size >= len(x):
                    end_batch = len(x)

                # Divide the train data into smaller batch
                current_x = x[start_batch:end_batch]
                current_y = y[start_batch:end_batch]
                
                print(f'current batch : {start_batch} - {end_batch}')

                # Train the neural network with current batch data
                for data in range(len(current_x)):
                    self.forward_propagation(current_x[data])
                    self.back_propagation(current_y[data])
                    self.partial_derivative(current_x[data])

                self.gradient_descent(current_x)

                # Move to next batch
                start_batch += self.batch_size
                end_batch += self.batch_size
        print('-------------------')
            

    def predict(self, x):
        '''
        Calculate the last layer's activation by applying forward propagation with input data

        Parameter :
            x : input data -> Numpy Array

        return :
            the last layer's activation -> Numpy Array
        '''
        self.forward_propagation(x)
        return self.layers[-1].activation[1:]


    def save(self, foldername):
        '''
        Save the weight of each layer to the foldername with .npy extension

        Parameter :
            foldername : foldername -> String
        '''
        import os
        try:
            if not os.path.exists(f'./{foldername}/'):
                os.makedirs(f'./{foldername}/')
        except:
            print(f'Error : cannot create {foldername}!')
        else:
            for index, layer in enumerate(self.layers):
                layer.save(f'{foldername}/{index}.npy')


    def load(self, foldername):
        '''
        Load the weight of each layer from the foldername with .npy extension

        Parameter :
            foldername : foldername -> String
        '''
        import os
        for index, layer in enumerate(self.layers):
            layer.load(f'{foldername}/{index}.npy')