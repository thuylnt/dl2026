import csv
import random
from Layer import Layer
from Loss import get_loss

def _minmax_apply(v, lo, hi):
    # Map raw value v into [0, 1]. If the column is constant (hi == lo) 
    # keep it at 0 instead of dividing by zero.
    return 0.0 if hi == lo else (v - lo) / (hi - lo)

def _minmax_invert(v, lo, hi):
    # Inverse of _minmax_apply: bring a [0, 1] value back to original units.
    return lo if hi == lo else v * (hi - lo) + lo

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        # populated by train() when normalize != 'none'
        self.input_scale = None   # list of (min, max) per input column
        self.output_scale = None  # list of (min, max) per output column

    def build_from_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        num_layers = int(lines[0])
        structure = [int(lines[i]) for i in range(1, num_layers + 1)]

        self.layers = []
        for i in range(num_layers):
            if i == 0:
                # input layer is a placeholder (no weighted neurons)
                self.layers.append(Layer(num_neurons=structure[i], input_size=0))
            else:
                self.layers.append(Layer(num_neurons=structure[i], input_size=structure[i - 1]))

    def load_parameters_from_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        index = 0
        for layer_id in range(1, len(self.layers)):
            for neuron in self.layers[layer_id].neurons:
                weights = [float(w) for w in lines[index].split(',')]
                bias = float(lines[index + 1])
                neuron.set_parameters(weights, bias)
                index += 2

    def save_parameters_to_file(self, file_path):
        with open(file_path, 'w') as f:
            for layer_id in range(1, len(self.layers)):
                for neuron in self.layers[layer_id].neurons:
                    f.write(', '.join(f"{w:.6f}" for w in neuron.weights) + '\n')
                    f.write(f"{neuron.bias:.6f}\n")

    def initialize_random_weights(self):
        for i in range(1, len(self.layers)):
            self.layers[i].initialize_random_weights()

    def forward(self, input_data):
        current = input_data
        for i in range(1, len(self.layers)):
            current = self.layers[i].forward(current)
        return current

    def predict(self, input_data):
        # Like forward(), but if train() was called with normalize='minmax',
        if self.input_scale is None:
            return self.forward(input_data)
        x_norm = [_minmax_apply(v, lo, hi)
                  for v, (lo, hi) in zip(input_data, self.input_scale)]
        y_norm = self.forward(x_norm)
        return [_minmax_invert(v, lo, hi)
                for v, (lo, hi) in zip(y_norm, self.output_scale)]

    def backward(self, d_outputs):
        # d_outputs[k] = dL/d(output_k), provided by the chosen loss function.
        upstream = list(d_outputs)
        for i in range(len(self.layers) - 1, 0, -1):
            upstream = self.layers[i].backward(upstream)

    def apply_gradients(self, learning_rate):
        for i in range(1, len(self.layers)):
            self.layers[i].apply_gradients(learning_rate)

    @staticmethod
    def _compute_scale(samples, col_start, col_end):
        # Per-column (min, max) over the specified slice of each row.
        scale = []
        for c in range(col_start, col_end):
            col = [row[c] for row in samples]
            scale.append((min(col), max(col)))
        return scale

    def _apply_scale(self, samples, num_inputs):
        # Return a new list of rows with inputs and outputs mapped to [0, 1]
        # using the scales we just computed. Originals are left untouched.
        out = []
        for row in samples:
            xs = [_minmax_apply(row[i], lo, hi)
                  for i, (lo, hi) in enumerate(self.input_scale)]
            ys = [_minmax_apply(row[num_inputs + i], lo, hi)
                  for i, (lo, hi) in enumerate(self.output_scale)]
            out.append(xs + ys)
        return out

    @staticmethod
    def _read_csv(file_path):
        # Row format: x1,...,xn,y1,...,ym. A non-numeric header row is skipped.
        samples = []
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                row = [c.strip() for c in row if c.strip() != '']
                if not row:
                    continue
                try:
                    samples.append([float(v) for v in row])
                except ValueError:
                    continue
        return samples

    def train(self, csv_path, learning_rate=0.5, epochs=10000,
              shuffle=True, verbose_every=1000, loss='mse',
              normalize='none'):
        loss_fn = get_loss(loss)

        samples = self._read_csv(csv_path)
        if not samples:
            raise ValueError(f"No usable samples found in {csv_path}")

        num_inputs = self.layers[1].input_size
        num_outputs = len(self.layers[-1].neurons)
        expected_cols = num_inputs + num_outputs
        for s in samples:
            if len(s) != expected_cols:
                raise ValueError(
                    f"Row has {len(s)} columns, expected {expected_cols} "
                    f"({num_inputs} inputs + {num_outputs} outputs)."
                )

        if normalize == 'minmax':
            self.input_scale = self._compute_scale(samples, 0, num_inputs)
            self.output_scale = self._compute_scale(samples, num_inputs, expected_cols)
            train_samples = self._apply_scale(samples, num_inputs)
            print(f"Normalization: minmax. input_scale={self.input_scale}, "
                  f"output_scale={self.output_scale}")
        elif normalize != 'none':
            raise ValueError(f"Unknown normalize mode: {normalize!r}")
        else:
            self.input_scale = None
            self.output_scale = None
            train_samples = samples

        print(f"Training with loss = {loss_fn.name.upper()}, "
              f"lr = {learning_rate}, epochs = {epochs}")

        for epoch in range(1, epochs + 1):
            if shuffle:
                random.shuffle(train_samples)

            epoch_loss = 0.0
            for row in train_samples:
                x = row[:num_inputs]
                y = row[num_inputs:]
                y_pred = self.forward(x)
                epoch_loss += loss_fn.forward(y_pred, y)
                d_outputs = loss_fn.backward(y_pred, y)
                self.backward(d_outputs)
                # stochastic gradient descent: update after every sample
                self.apply_gradients(learning_rate)

            if verbose_every and (epoch == 1 or epoch % verbose_every == 0 or epoch == epochs):
                print(f"Epoch {epoch:>6d} / {epochs} - loss: {epoch_loss:.6f}")

        return samples

    def print_structure(self):
        print("Neural Network Structure:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: {len(layer.neurons)} neurons")
            for j, neuron in enumerate(layer.neurons):
                if i == 0:
                    print(f"  Neuron {j + 1}: (input placeholder)")
                else:
                    weights = ", ".join(f"{w:.4f}" for w in neuron.weights)
                    print(f"  Neuron {j + 1}: Weights: [{weights}], Bias: {neuron.bias:.4f}")
