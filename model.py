import numpy as np

def sigmoid_function(x):
    if x == 0:
        return 0.5
    if x < 0:
        return np.exp(x) / (1 + np.exp(x))
    else:
        return 1 / (1 + np.exp(-x))

sigmoid = np.vectorize(sigmoid_function)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Dense_Layer:
    def __init__(self, n, m, a='sigmoid'):
        self.no_of_input = n
        self.no_of_neuron = m
        self.weights = np.random.randn(m, n+1).astype(np.float64)# Add the biases in weights by adding 1 in input
        self.activation = a
        self.output = None
        self.input = None

    def activate(self, neuron_input):
        self.input = neuron_input.astype(np.float64)
        x= np.vstack((self.input, np.ones((1,self.input.shape[1])) ))
        y = np.dot(self.weights, x) # Using the formulas : weights dot input augmented by ones
        self.combination = y

        if self.activation == 'sigmoid':
            self.output = np.vstack((sigmoid(y), np.ones((1,sigmoid(y).shape[1])) ))
        elif self.activation == 'lu':
            self.output = np.vstack((y, np.ones((1,y.shape[1])) ))
        elif self.activation == 'relu':
            self.output =np.vstack((np.maximum(0, y), np.ones((1,np.maximum(0, y).shape[1])) ))
            

class Model:
    def __init__(self, list_of_dense_layers):
        self.layers = list_of_dense_layers
        self.outputs = None
        self.___task_type = 'regression' # or classification

"""
Here, we shall use the F-adjoint formulation (A new alternative to the backpropagation method) introduced in my recent arxiv preprint: https://arxiv.org/abs/2304.13820
"""
    def F_propagation(self, model_input):
        model_input = model_input.astype(np.float64)
        for layer in self.layers:
            layer.activate(model_input)
            model_input = layer.output
        self.outputs = self.layers[-1].output[:,:-1] # We should  dellete tne added ones from the output layer activation.
        return self.outputs

    def mean_square_error(self, y):
        squared_diff = (y - self.outputs) ** 2
        return np.mean(squared_diff)

    def Fstar_propagation(self, y, alpha=0.01):
        y = y.astype(np.float64)
        alpha = np.float64(alpha)
        error = self.outputs - y

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            if layer.activation == 'sigmoid':
                xstar = error
                ystar=xstar * sigmoid_derivative(layer.combination)
            elif layer.activation == 'relu':
                delta = error * (layer.combination > 0).astype(np.float64)
            else:  # for linear activation 'lu'
                delta = error

            if i > 0:
                prev_output = self.layers[i - 1].output
            else:
                prev_output = self.layers[0].input

            d_weights = np.dot(prev_output.T, delta)
            d_biases = np.sum(delta, axis=0)

            layer.weights -= alpha * d_weights
            layer.biases -= alpha * d_biases

            error = np.dot(delta, layer.weights.T)

    def fit(self, X_train, y_train, learning_rate=0.01, epoch=10):
        for epok in range(epoch):
            self.propagate(X_train)
            self.back_propagate(y_train, alpha=learning_rate)
            # logging
            if self.___task_type == 'classification':
                print(f'epoch({epok}) complete  | error = {self.log_loss_error(y_train)}')
            else:
                print(f'epoch({epok}) complete  | error = {self.mean_square_error(y_train)}')

# ========================================================================================
# ========================================================================================
