import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


input = np.arange(-10, 10, 0.1)
output = sigmoid(input)

fig = plt.figure()
plt.plot(input, output)
plt.show()