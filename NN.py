import numpy as np
import random


def activate_function(x):
    return 1/(1 + pow(2.716, -x))


def create_weight(layers_num, neurons_num):
    a = [[] for _ in range(layers_num)]
    for b in range(layers_num - 1):
        for g in range(neurons_num[b]):
            c = 2 * np.random.random(neurons_num[b + 1]) - 1
            a[b].append(c)
    return a


def create_sample_out(layers_num, neurons_num):
    a = [[] for _ in range(layers_num)]
    for r in range(layers_num):
        a[r] = [0.0 for _ in range(neurons_num[r])]
    return a


def create_sample_last_change(layers_num, neurons_num):
    a = [[] for _ in range(layers_num)]
    for r in range(layers_num - 1):
        for y in range(neurons_num[r]):
            a[r].append([0 for _ in range(neurons_num[r + 1])])
    return a


class NeuralNetwork:
    def __init__(self, neurons_num):
        self.layers_num = len(neurons_num)
        self.neurons_num = np.array(neurons_num)
        self.weights = create_weight(self.layers_num, self.neurons_num)
        self.outs = create_sample_out(self.layers_num, self.neurons_num)
        self.momentum = 0.1
        self.learning_speed = 0.3

    def teach(self, tests, ans, epochs, show_progress=False):
        for epoch in range(epochs):
            if show_progress:
                print(epoch, 'of', epochs)
            tests_num = len(tests)
            last_change = create_sample_last_change(self.layers_num, self.neurons_num)
            for test_num in range(tests_num):
                self.outs = create_sample_out(self.layers_num, self.neurons_num)
                self.outs[0] = tests[test_num]

                # creates outputs for hidden layer neurons
                for hidden_layer in range(1, self.layers_num):
                    for hidden_neuron in range(self.neurons_num[hidden_layer]):
                        s = 0
                        for last_layer_neuron in range(self.neurons_num[hidden_layer - 1]):
                            s += self.weights[hidden_layer - 1][last_layer_neuron][hidden_neuron] *\
                                 self.outs[hidden_layer - 1][last_layer_neuron]
                        self.outs[hidden_layer][hidden_neuron] = activate_function(s)

                # creates list of deltas for each layer
                errors = []
                for out in range(len(self.outs[self.layers_num - 1])):
                    error = ans[test_num][out] - self.outs[self.layers_num - 1][out]
                    errors.append(error)
                output_delta = []
                for out in range(len(self.outs[self.layers_num - 1])):
                    new_delta = errors[out]*(1 - self.outs[self.layers_num - 1][out]) *\
                                self.outs[self.layers_num - 1][out]
                    output_delta.append(new_delta)
                deltas = [[] for _ in range(self.layers_num - 1)] + [output_delta]
                for layer in range(self.layers_num - 2, 0, -1):
                    for neuron in range(self.neurons_num[layer]):
                        delta = self.outs[layer][neuron] * (1 - self.outs[layer][neuron]) *\
                                (self.weights[layer][neuron] * deltas[layer + 1]).sum()
                        deltas[layer].append(delta)

                # updates weights
                for layer in range(self.layers_num - 2, -1, -1):
                    for neuron in range(self.neurons_num[layer]):
                        for next_neuron in range(self.neurons_num[layer + 1]):
                            gradient = deltas[layer + 1][next_neuron] * self.outs[layer][neuron]
                            change = gradient*self.learning_speed+last_change[layer][neuron][next_neuron]*self.momentum
                            last_change[layer][neuron][next_neuron] = change
                            self.weights[layer][neuron][next_neuron] += change

    def predict_answer(self, inp):
        self.outs = create_sample_out(self.layers_num, self.neurons_num)
        self.outs[0] = inp
        for hidden_layer in range(1, self.layers_num):
            for hidden_neuron in range(self.neurons_num[hidden_layer]):
                s = 0
                for last_layer_neuron in range(self.neurons_num[hidden_layer - 1]):
                    s += self.weights[hidden_layer - 1][last_layer_neuron][hidden_neuron] * \
                         self.outs[hidden_layer - 1][last_layer_neuron]
                self.outs[hidden_layer][hidden_neuron] = activate_function(s)
        return self.outs[self.layers_num - 1]

