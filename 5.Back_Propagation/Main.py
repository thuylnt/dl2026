import random
import sys

from NeuralNetwork import NeuralNetwork

def main():
    # CLI: python Main.py [csv_path] [loss] [structure_path] [output_params_path] [normalize]
    #   loss     : "mse" or "bce"
    #   normalize: "none" (default) or "minmax" (needed for raw-scale data e.g. house price)
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'xor.csv'
    loss = sys.argv[2] if len(sys.argv) > 2 else 'bce'
    structure_path = sys.argv[3] if len(sys.argv) > 3 else 'structure.txt'
    output_params = sys.argv[4] if len(sys.argv) > 4 else 'trained_parameters.txt'
    normalize = sys.argv[5] if len(sys.argv) > 5 else 'none'

    random.seed(42)  # reproducible runs

    print("Building neural network from", structure_path)
    nn = NeuralNetwork()
    nn.build_from_file(structure_path)

    print("Initializing random weights...")
    nn.initialize_random_weights()
    nn.print_structure()

    # BCE gradients lack the o(1-o) attenuation -> roughly 4x bigger than MSE.
    # Use a smaller default lr for BCE to avoid landing in a saddle.
    learning_rate = 0.1 if loss.lower() == 'bce' else 0.5

    print(f"\nTraining on {csv_path} with backpropagation + gradient descent...")
    samples = nn.train(csv_path, learning_rate=learning_rate, epochs=10000,
                       shuffle=True, verbose_every=1000, loss=loss,
                       normalize=normalize)

    print("\nFinal parameters:")
    nn.print_structure()

    nn.save_parameters_to_file(output_params)
    print(f"\nSaved trained parameters to {output_params}")

    num_inputs = nn.layers[1].input_size
    print("\nEvaluation on training set:")
    # predict() handles normalize/denormalize automatically if train ran with normalize='minmax', 
    # so raw-unit x and y_pred print correctly either way.
    for row in samples:
        x = row[:num_inputs]
        y = row[num_inputs:]
        y_pred = nn.predict(x)
        rounded = [round(v) for v in y_pred]
        print(f"  input={x}  target={y}  predicted={[f'{v:.4f}' for v in y_pred]}  -> {rounded}")

if __name__ == '__main__':
    main()
