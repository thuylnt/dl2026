from Neuron import Neuron

class Layer:
    def __init__(self, num_neurons, input_size):
        self.input_size = input_size
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def initialize_random_weights(self):
        for neuron in self.neurons:
            neuron.initialize_random_weights()

    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]

    def backward(self, d_outputs):
        # d_outputs[i] = dL/d(output of neuron i)
        # returns dL/d(input) summed across neurons (upstream gradient)
        upstream = [0.0 for _ in range(self.input_size)]
        for neuron, d_out in zip(self.neurons, d_outputs):
            grads_in = neuron.backward(d_out)
            for j in range(self.input_size):
                upstream[j] += grads_in[j]
        return upstream

    def apply_gradients(self, learning_rate):
        for neuron in self.neurons:
            neuron.apply_gradients(learning_rate)
