import numpy as np


def add_forward(a, b):
    return a + b

def mul_forward(a, b):
    return a * b

def log_forward(a):
    return np.log(a)

def exp_forward(a):
    return np.exp(a)

def neg_forward(a):
    return -a

def pow_forward(a, exp):
    return a ** exp

def relu_forward(a):
    return np.maximum(a, 0)

def sigmoid_forward(a):
    return 1.0 / (1.0 + np.exp(-a))

def tanh_forward(a):
    return np.tanh(a)