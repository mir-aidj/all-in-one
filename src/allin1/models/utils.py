import torch.nn as nn


def get_activation_function(name: str):
  activation_functions = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'softmax': nn.Softmax(dim=1),
    'log_softmax': nn.LogSoftmax(dim=1),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'gelu': nn.GELU(),
    'prelu': nn.PReLU(),
  }
  
  if name in activation_functions:
    return activation_functions[name]
  else:
    raise ValueError(f"Unsupported activation function: {name}")
