import sqlite3 as sl
import pickle
import numpy as np
import math
from test_data import TestData

# weights = [np.random.randn(150, 784) * 0.01, np.random.randn(150, 150) * 0.01, np.random.randn(10, 150) * 0.01]
# biases = [np.zeros((150, 1)), np.zeros((150, 1)), np.zeros((10, 1))]

weights = pickle.load(open("../net_config/weights.p", "rb"))
biases = pickle.load(open("../net_config/biases.p", "rb"))

data = TestData()

learning_rate = 0.0001


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

def main():
    incorrect = 0
    for i in range(1, 10001):
        image = data.get_image(i)
        image_label = image[2]
        image_array = pickle.loads(image[1])

        layer_output = [image_array.reshape(-1, 1) / 255]


        # activation = np.matmul(weights[0], layer_output[0])
        # print(activation)
        for i in range(3):
            z = np.matmul(weights[i], (layer_output[i])) + biases[i]
            layer_output.append(sigmoid(z))

        # print(layer_output[-1])
        result = np.where(layer_output[-1] == np.amax(layer_output[-1]))
        print("Predicted: " + str(result[0][0]))
        print("Actual Number: " + str(image_label))

        if result[0][0] != image_label:
            incorrect += 1
    print("Error rate of " + str(incorrect / 10000))

main()