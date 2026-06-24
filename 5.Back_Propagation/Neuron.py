import random
import math

class Neuron:
    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = [0.0 for _ in range(num_inputs)]
        self.bias = 0.0
        self.last_inputs = []
        self.output = 0.0
        self.grad_weights = [0.0 for _ in range(num_inputs)]
        self.grad_bias = 0.0

    def initialize_random_weights(self):
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(self.num_inputs)]
        self.bias = random.uniform(-1.0, 1.0)

    def set_parameters(self, weights, bias):
        self.weights = list(weights)
        self.bias = bias

    def activate(self, inputs):
        self.last_inputs = list(inputs)
        total = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.sigmoid(total)
        return self.output

    def sigmoid(self, x):
       return 1 / (1 + math.exp(-x))

    def backward(self, d_output):
        d_z = d_output * self.output * (1.0 - self.output)
        for i in range(self.num_inputs):
            self.grad_weights[i] += d_z * self.last_inputs[i]
        self.grad_bias += d_z
        return [d_z * w for w in self.weights]

    def apply_gradients(self, learning_rate):
        for i in range(self.num_inputs):
            self.weights[i] -= learning_rate * self.grad_weights[i]
            self.grad_weights[i] = 0.0
        self.bias -= learning_rate * self.grad_bias
        self.grad_bias = 0.0
