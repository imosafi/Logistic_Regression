import numpy as np
import random
import matplotlib.pyplot as plot
import math


ALPHA = 0.1
CLASSES = 3
D = 1
EPOCHS = 150
PRINT_ACCURACY = False


def calc_softmax(x):
    # numerically stable softmax calculation
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def generate_dataset():
    dataset = []
    for i in xrange(1, 4):
        for val in np.random.normal(2 * i, 1, 100):
            y = [0] * CLASSES
            y[i - 1] = 1
            dataset.append([val, y])
    return dataset


def calc_prediction(x, W, b):
    return calc_softmax(np.transpose(W) * x + b)


def calc_grad(x, y, y_pred):
    gw = (x * (y_pred - y))
    gb = (y_pred - y)

    return gw, gb


def get_real_x_probability(x):
    a = calc_noraml_prob(x, 2, 1)
    return a / (a + calc_noraml_prob(x, 4, 1) + calc_noraml_prob(x, 6, 1))


def calc_noraml_prob(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def create_plot(dataset, W, b):
    plot.ylabel('p(y=1|x)')
    plot.xlabel('x')
    plot.xlim(0, 10)
    # x_1_values = [x[0] for x in dataset if np.argmax(x[1]) == 0]
    # x_values = [x[0] for x in dataset]
    x_values = np.arange(0, 10, 0.05)
    model_prob_x_1 = [calc_prediction(x, W, b)[0] for x in x_values]
    plot.scatter(x_values, model_prob_x_1, marker='.', color='b')

    real_prob_x1 = [get_real_x_probability(x) for x in x_values]
    plot.scatter(x_values, real_prob_x1, marker='.', color='r')

    plot.show()


def main():
    dataset = generate_dataset()
    W = [0] * 3
    b = [0] * 3

    for i in xrange(EPOCHS):
        acc = 0.0
        random.shuffle(dataset)
        for x, y in dataset:
            y_pred = calc_prediction(x, W, b)
            if np.argmax(y_pred) == np.argmax(y):
                acc += 1
            gw, gb = calc_grad(x, y, y_pred)
            W = W - ALPHA * gw
            b = b - ALPHA * gb
        if PRINT_ACCURACY:
            acc /= len(dataset)
            print 'Prediction accuracy is: ' + str(acc)

    create_plot(dataset, W, b)


if __name__ == '__main__':
    main()
