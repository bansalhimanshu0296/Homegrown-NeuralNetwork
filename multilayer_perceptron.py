# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: Himanshu Himanshu -- hhimansh
#
# Based on skeleton code by CSCI-B 551 Fall 2021 Course Staff
#
# Reference for backpropogation taken from Andrew ng deep learning course from coursera and notes: https://www.dropbox.com/s/nfv5w68c6ocvjqf/0-2.pdf?dl=0

import numpy as np
from utils import identity, sigmoid, tanh, relu, softmax, cross_entropy, one_hot_encoding

class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        
        # setting class variable setting _X as training data and _y as one hot encoding of labels
        self._X = X
        self._y = one_hot_encoding(y)

        # getting index of each category from one hot encoded categories
        self.categories = {}
        for i in range(len(y)):
            self.categories[y[i]] = np.argmax(self._y[i])
        np.random.seed(42)
        
        # Initialising class variables _h_weights, _h_bias, _o_weights, _o_bias as random variables
        self._h_weights = np.random.random((len(self._X[0]), self.n_hidden))
        self._h_bias = np.random.random((1, self.n_hidden))
        self._o_weights = np.random.random((self.n_hidden, len(self._y[0])))
        self._o_bias = np.random.random((1, len(self._y[0])))


    # Reference for backpropogation taken from Andrew ng deep learning course from coursera and notes: https://www.dropbox.com/s/nfv5w68c6ocvjqf/0-2.pdf?dl=0
    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        # Calling initaialize function to set different class variables
        self._initialize(X, y)

        # Iterating loop for n_iterations time to get weights
        for i in range(self.n_iterations):

            # Forward Propogation Process
            # multiplying input weights i.e. _h_weights and adding bias i.e _o_bias to input i.e. training data
            input_output  =  np.dot(self._X, self._h_weights) + self._h_bias

            # getting hidden layer output by applying activation function of hidden layer to input layer output 
            # and multiplying it by output weights i.e. _o_weights and adding output bias i.e. _o_bias
            hidden_output = np.dot(self.hidden_activation(input_output), self._o_weights) + self._o_bias
            
            # finding final probabilities of each category by applying output layer activation function to get the true 
            # output that can be used for calculating loss
            output =  self._output_activation(hidden_output)

            # starting back propogation process for calculating weight and bias error for each layer
            
            # Caculating output layer error by just subtracting actual y from predicted y
            output_layer_error = hidden_output - self._y 

            # calculating weight error for output layer by multiplication output_layer_error transpose to
            # hidden layer activation on input layer output and taking transpose of whole thing and we are taking mean of error by 
            # dividing it by numner of samples
            output_weight_error = (1/len(self._X)) * np.dot(output_layer_error.T, self.hidden_activation(input_output)).T

            # Calculating bias error for output layer sum of output_layer_error for each feature axis and taking mean of it
            output_bias_error = (1/len(self._X)) * np.sum(output_layer_error, axis = 0, keepdims = True)

            # Caculating input layer error by weights of output layer to output layer error transpose and taking taking transpose of whole
            # and projecting it to derivative of hidden layer activation 
            input_layer_error = np.dot(self._o_weights, output_layer_error.T).T * self.hidden_activation(input_output, derivative = True)

            # Calculating input weight layer error by multiplying transpose of input layer error matrix and input data 
            # and taking transpose of whole matrix and then taking mean for each cell
            input_weight_error = (1/len(self._X)) * np.dot(input_layer_error.T, self._X).T

            # Calculating bias error for input layer sum of input_layer_error for each feature axis and taking mean of it
            input_bias_error = (1/len(self._X)) * np.sum(input_layer_error, axis = 0, keepdims = True)

            # Adjusting weights of output layer and input layer and bias of input and output 
            # layer by multiplying error for respective with learning rate and subtracting it from current value
            # we are subtracting because we take negative gradient descent while calculating new weights and bias
            self._o_weights -= self.learning_rate * output_weight_error
            self._o_bias -= self.learning_rate * output_bias_error
            self._h_weights -= self.learning_rate * input_weight_error
            self._h_bias -= self.learning_rate * input_bias_error

            # after every 20 iterations calculating history loss by calling loss function on output probabilities 
            # which can be calculated by applying output activation function on output layer output and actual output
            if i%20 == 0:
                self._loss_history.append(self._loss_function(self._y, output))
       
    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """

        # taking all the keys i.e. categories from dictionary
        category_key = list(self.categories.keys())

        # taking all the index of categories from dictionary
        category_index = list(self.categories.values())

        # Initialising empty list for predicted y
        predicted_y = []

        # Iterating through every point in test data
        for X_test in X:

            # Caculating after going through each layer of preceptron initially giving data to input layer 
            # where it is multiplied by input weights and then input bias is added to it
            input_output  =  np.dot(X_test, self._h_weights) + self._h_bias

            # Calculating hidden layer output by calling hidden layer activation out input layer output 
            # and multiplying it with output layer weights and adding output layer bias to it and it is final output
            hidden_output = np.dot(self.hidden_activation(input_output), self._o_weights) + self._o_bias

            # calculating output probability of each class from output by calling output layer activation function i.e. softmax
            output =  self._output_activation(hidden_output)

            # appending category which have the max probability by find the max from output and finding its index in index array and 
            # then finding corresponding category
            predicted_y.append(category_key[category_index.index(np.argmax(output))])

        # returning predicted y list
        return predicted_y