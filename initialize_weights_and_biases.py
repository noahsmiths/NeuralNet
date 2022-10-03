import numpy as np
import pickle


weights = [np.random.randn(150, 784) * 0.01, np.random.randn(150, 150) * 0.01, np.random.randn(10, 150) * 0.01]
biases = [np.zeros((150, 1)), np.zeros((150, 1)), np.zeros((10, 1))]

pickle.dump(weights, open("net_config/weights.p", "wb"))
pickle.dump(biases, open("net_config/biases.p", "wb"))
