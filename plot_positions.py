import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#load positions from file
positions_raw = pd.read_csv('positions.csv')
# positions_filtered = pd.read_csv('exponential_filtered_positions.csv')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.text(positions_raw['x'][0], positions_raw['z'][0], positions_raw['y'][0], 'Start')
ax.scatter(positions_raw['x'], positions_raw['z'], positions_raw['y'], c=positions_raw['color'], marker='o')
ax.text(positions_raw['x'][len(positions_raw['x'])-1], positions_raw['z'][len(positions_raw['z'])-1], positions_raw['y'][len(positions_raw['y'])-1], 'End')


ax.set_xlabel('X [cm]')
ax.set_ylabel('Y [cm]')
ax.set_zlabel('Z [cm]')

plt.show()