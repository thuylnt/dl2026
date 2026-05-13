from Neuron import Neuron

class Layer:
    def __init__(self, num_neurons, input_size):
        self.neurons = [Neuron(input_size) for _ in range(num_neurons)]

    def initialize_random_weights(self):
        for neuron in self.neurons:
            neuron.initialize_random_weights()

    def forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            output = neuron.activate(inputs)
            outputs.append(output)
        print("(Debug) Layer forward outputs: ", outputs)
        return outputs