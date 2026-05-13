from NeuralNetwork import NeuralNetwork
from Neuron import Neuron
from Layer import Layer

print("Building neural network from file...")
# nn = NeuralNetwork()
# nn.build_from_file('structure.txt')
# print("Initializing random weights...")
# nn.initialize_random_weights()
# nn.print_structure()
# input_data = [0.5, 0.3]  # Example input
# output = nn.forward(input_data)
# print("Output of the neural network:", output)

print("-" * 50)
nn1 = NeuralNetwork()
nn1.build_from_file('structure.txt')
print("\nLoading parameters from file...")
nn1.load_parameters_from_file('parameters.txt')
nn1.print_structure()

input_data =  [[0,0], [0,1], [1,0], [1,1]]  # Example input
for i, data in enumerate(input_data):
    print(f"\nInput {i + 1}: {data}")
    output = nn1.forward(data)
    print("Output of the neural network:", output)
