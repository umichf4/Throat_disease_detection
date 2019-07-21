import numpy as np
import matplotlib.pyplot as plt

# y = mx + b
# m is slope, b is y-intercept


def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    pre_error = compute_error_for_line_given_points(b, m, points)
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
        if i % 100 == 0:
            cur_error = compute_error_for_line_given_points(b, m, points)
            if cur_error > 0.995 * pre_error:
                break
            else:
                pre_error = cur_error
    return [b, m, i]


def plot_pic(points, b, m):
    x = points[:, 0]
    y = points[:, 1]
    start = min(x)
    end = max(x)
    predict_start = start * m + b
    predict_end = end * m + b
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.scatter(x, y, color='blue', label="original data")
    plt.plot([start, end], [predict_start, predict_end], color='red', label="regression curve")
    plt.show()


def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def run():
    mask = np.load("mask_rotated_135.npy")
    index = np.nonzero(mask)
    points = normalize(np.column_stack((index[1], index[0])))
    learning_rate = 0.01
    initial_b = 0  # initial y-intercept guess
    initial_m = 1  # initial slope guess
    num_iterations = 5000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m, i] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}"
          .format(i, b, m, compute_error_for_line_given_points(b, m, points)))
    plot_pic(points, b, m)


if __name__ == '__main__':
    run()