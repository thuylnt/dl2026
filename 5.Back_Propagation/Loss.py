import math

# Clip outputs away from {0, 1} so log() in BCE never blows up.
_EPS = 1e-12

def _clip(x):
    if x < _EPS:
        return _EPS
    if x > 1.0 - _EPS:
        return 1.0 - _EPS
    return x

class MSELoss:
    """Mean squared error: L = 0.5 * sum_k (o_k - t_k)^2."""

    name = 'mse'

    def forward(self, outputs, targets):
        return 0.5 * sum((o - t) ** 2 for o, t in zip(outputs, targets))

    def backward(self, outputs, targets):
        # dL/d(o_k) = o_k - t_k
        return [o - t for o, t in zip(outputs, targets)]


class BCELoss:
    """Binary cross entropy: L = -sum_k [ t_k log(o_k) + (1 - t_k) log(1 - o_k) ].

    Assumes outputs are sigmoid (in (0, 1)) and targets are in {0, 1}.
    """

    name = 'bce'

    def forward(self, outputs, targets):
        total = 0.0
        for o, t in zip(outputs, targets):
            oc = _clip(o)
            total -= t * math.log(oc) + (1.0 - t) * math.log(1.0 - oc)
        return total

    def backward(self, outputs, targets):
        # dL/d(o_k) = (o_k - t_k) / (o_k (1 - o_k))
        # Combined with sigmoid derivative o(1-o) inside the neuron
        grads = []
        for o, t in zip(outputs, targets):
            oc = _clip(o)
            grads.append((oc - t) / (oc * (1.0 - oc)))
        return grads


def get_loss(name):
    key = (name or '').lower()
    if key == 'mse':
        return MSELoss()
    if key == 'bce':
        return BCELoss()
    raise ValueError(f"Unknown loss '{name}'. Use 'mse' or 'bce'.")
