import sqlite3 as sl
import pickle
import numpy as np
import math
from training_data import TrainingData

# weights = [np.random.randn(150, 784) * 0.01, np.random.randn(150, 150) * 0.01, np.random.randn(10, 150) * 0.01]
# biases = [np.zeros((150, 1)), np.zeros((150, 1)), np.zeros((10, 1))]

weights = pickle.load(open("net_config/weights.p", "rb"))
biases = pickle.load(open("net_config/biases.p", "rb"))

data = TrainingData()

learning_rate = 0.1


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return x * (1 - x)


def cost(output, expected):
    return (1 / 2) * ((output - expected)**2)


def cost_deriv(output, expected):
    return output - expected


def total_cost(output, expected):
    total = 0
    for i in range(10):
        total += cost(output[i], 1 if i == expected else 0)

    return total


def main(k):
    image = data.get_image(k)
    image_label = image[2]
    image_array = pickle.loads(image[1])

    layer_output = [image_array.reshape(-1, 1) / 255]

    # print(layer_output[0])

    # activation = np.matmul(weights[0], layer_output[0])
    # print(activation)
    for i in range(3):
        z = np.matmul(weights[i], layer_output[i]) + biases[i]

        layer_output.append(sigmoid(z))

    print(total_cost(layer_output[-1], image_label)) # Print total cost

    output_layer_error = np.zeros((10, 1))
    output_layer_error[image_label] = 1

    output_layer_error = cost_deriv(layer_output[-1], output_layer_error)

    # return print(output_layer_error)

    # print(output_layer_error)

    dZ3 = np.multiply(output_layer_error, sigmoid_prime(layer_output[3]))
    dW3 = np.matmul(dZ3, layer_output[2].T)
    dB3 = dZ3


    dZ3_back = np.matmul(dZ3.T, weights[2]).T
    dZ2 = np.multiply(dZ3_back, sigmoid_prime(layer_output[2]))
    dW2 = np.matmul(dZ2, layer_output[1].T)
    dB2 = dZ2


    dZ2_back = np.matmul(dZ2.T, weights[1]).T
    dZ1 = np.multiply(dZ2_back, sigmoid_prime(layer_output[1]))
    dW1 = np.matmul(dZ1, layer_output[0].T)
    dB1 = dZ1

    weights[2] = weights[2] - learning_rate * dW3
    biases[2] = biases[2] - learning_rate * dB3

    weights[1] = weights[1] - learning_rate * dW2
    biases[1] = biases[1] - learning_rate * dB2

    weights[0] = weights[0] - learning_rate * dW1
    biases[0] = biases[0] - learning_rate * dB1

    # for i in range(1, 4):

    # print(output_layer_error)
    # print(layer_output)
for m in range(0, 10):
    for k in range(1, 60001):
        print(k)
        main(k)

pickle.dump(weights, open("net_config/weights.p", "wb"))
pickle.dump(biases, open("net_config/biases.p", "wb"))