import numpy as np
import matplotlib.pyplot as plt

def naive_rolling_average(data, window_size):
    num_rows, num_columns = data.shape
    filtered_data = np.zeros((num_rows - window_size + 1, num_columns))

    for i in range(num_rows - window_size + 1):
        window = data[i:i + window_size]
        filtered_data[i] = np.mean(window, axis=0)

    return filtered_data

def rolling_average(data, window_size):
    cumsum = np.cumsum(data, axis = 0)
    cumsum[window_size:] = cumsum[:-window_size] - cumsum[window_size - 1:-1]
    return cumsum[window_size - 1:] / window_size

def exponential_moving_average(data, alpha):
    filtered_data = np.zeros_like(data)
    filtered_data[0] = data[0]

    for i in range(1, len(data)):
        filtered_data[i] = alpha * data[i] + (1 - alpha) * filtered_data[i - 1]

    return filtered_data

# Generate some example data (replace this with your real tracker data)
# data = np.random.randn(100, 3) * np.array([1, 1, 10]) + np.array([0, 0, 100])
#Generate 3d sine wave with noise added
num_points = 100
noise_std_dev = 0.1
t = np.linspace(0, 4*np.pi, num_points)

x = np.sin(t)
y = np.cos(t)
z = t

noise = np.random.normal(0, noise_std_dev, (num_points,3))
x_noise=x+noise[:,0]
y_noise = y + noise[:, 1]
z_noise = z + noise[:, 2]

data = np.array([x, y, z]).T
data_noisy = np.array([x_noise, y_noise, z_noise]).T

# Apply the naive rolling average filter
window_size = 5
filtered_data = exponential_moving_average(data, .5)

#create 3d plot
ax = plt.axes(projection='3d')

#plot the data
ax.plot(data[:, 0], data[:, 1], data[:, 2], c='b', label='raw data')
ax.scatter(filtered_data[:, 0], filtered_data[:, 1], filtered_data[:, 2], c='r', label='filtered data')
ax.scatter(data_noisy[:, 0], data_noisy[:, 1], data_noisy[:, 2], c='g', label='noisy data')

plt.show()

print("Filtered data:", filtered_data)