import random
import math

class Neuron :
    def __init__(self, num_inputs):
        self.weights = [0.0 for _ in range(num_inputs)]
        self.bias = 0.0
        self.output = 0.0

    def initialize_random_weights(self):
        self.weights = [random.random() for _ in range(len(self.weights))]
        self.bias = random.random()

    def set_parameters(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        #print("Activating neuron with inputs: ", inputs)
        #if len(self.weights) == 0:
        #    return inputs  # For input neurons, output is the input itself
        
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.sigmoid(total)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))