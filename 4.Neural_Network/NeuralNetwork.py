from Layer import Layer
class NeuralNetwork:
    def __init__(self):
        self.layers = []
    
    def build_from_file(self, file_path):
        # Implement logic to build the neural network from a file
        with open(file_path, 'r') as f: 
            # Read and parse the file to construct layers and neurons
            lines = f.readlines()
        lines = [line.strip() for line in lines]    
        num_layers = int(lines[0])
        structure = []
        for i in range(1, num_layers + 1):            
            structure.append(int(lines[i]))
        
        for i in range(num_layers):
            if i == 0:
                self.layers.append(Layer(num_neurons=structure[i], input_size=0))  # Input layer
            else:
                self.layers.append(Layer(num_neurons=structure[i], input_size=structure[i-1]))  
                # Hidden and output layers

    def load_parameters_from_file(self, file_path):
        # Implement logic to load weights and biases from a file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        
        index = 0
        
        for layer_id in range(1, len(self.layers)):

            layer = self.layers[layer_id] 

            for neuron in layer.neurons:
                weights_line = lines[index].split(',')
                bias_line = lines[index + 1]
                weights = list(map(float, weights_line))
                bias = float(bias_line)
                neuron.set_parameters(weights, bias)
                index += 2
        
    def initialize_random_weights(self):
        # Skip input layer
        for i in range(1, len(self.layers)):
            self.layers[i].initialize_random_weights()

    def forward(self, input_data):
        current_outputs = input_data    
        # Skip input layer
        for i in range(1, len(self.layers)):
            current_outputs = self.layers[i].forward(current_outputs)
        return current_outputs

    def print_structure(self):
        print("Neural Network Structure:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: {len(layer.neurons)} neurons")
            for j, neuron in enumerate(layer.neurons):
                print(f"  Neuron {j + 1}: Weights: {neuron.weights}, Bias: {neuron.bias}")
