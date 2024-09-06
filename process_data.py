import numpy as np
import pandas as pd

def inter_frame_correlation(positions):
    raw_positions = positions[['x', 'y', 'z', 'frame_number', 'delta_time']].values

    prev_positions = raw_positions[0]
    inst_velocity = np.linalg.norm(raw_positions[1][0:3] - prev_positions[0:3])/raw_positions[1][4]
    prev_positions = raw_positions[1]
    prev_inst_velocity = inst_velocity

    avg_xz_velocity = np.linalg.norm(np.array([raw_positions[1][0], raw_positions[1][2]]) - np.array([raw_positions[0][0], raw_positions[0][2]]))/raw_positions[1][4]

    for point in raw_positions[2:]:
        inst_velocity = np.linalg.norm(point[0:3] - prev_positions[0:3])/point[4]
        avg_xz_velocity += np.linalg.norm(np.array([point[0], point[2]]) - np.array([prev_positions[0], prev_positions[2]]))/point[4]


        if abs(point[2] - prev_positions[2]) > .5:
            print(f'\nFrame {point[3]} has a large change in z position!!!')
        
        prev_positions = point
        prev_inst_velocity = inst_velocity

    avg_xz_velocity /= len(raw_positions)
    print(f'Average XZ velocity: {avg_xz_velocity}')

def apply_kalman_filter(positions):
    pass

def apply_moving_average(positions, window_size):
    """Apply a moving average to the positions to smooth out the data.

    Args:
        positions (DataFrame): Data to be smoothed
        window_size (int): Number of neighboring points to average over

    Returns:
        rolling_average_smoothed_df (DataFrame): The smoothed data
    """
    
    x = positions['x']
    y = positions['y']
    z = positions['z']
    #compile into a single array
    points = np.array([x, y, z])
    
    #Apply moving average
    cumulative_sum = np.cumsum(points, axis=0)
    cumulative_sum[window_size:] = cumulative_sum[:-window_size] - cumulative_sum[window_size -1:-1]
    rolling_average_smoothed = cumulative_sum[window_size - 1:] / window_size
    
    #save back into dataframe
    rolling_average_smoothed_df = pd.DataFrame(data=rolling_average_smoothed.T, columns=['x', 'y', 'z'])
    rolling_average_smoothed_df.to_csv("rolling_average_smoothed.csv")
    
    print(rolling_average_smoothed_df)
    return rolling_average_smoothed_df

def apply_exponential_moving_average(positions, alpha=.2):
    points = np.stack((positions['x'], positions['y'], positions['z']), axis =1)
    filtered_points = np.zeros_like(points)
    filtered_points[0] = points[0]
    
    for i in range(1, len(points)):
        #apply exponential moving average

        filtered_points[i][0] = alpha * points[i][0] + (1 - alpha) * filtered_points[i-1][0]
        filtered_points[i][1] = alpha * points[i][1] + (1 - alpha) * filtered_points[i-1][1]
        filtered_points[i][2] = (alpha/2) * points[i][2] + (1 - (alpha/2)) * filtered_points[i-1][2] #apply twice the smoothing to the z axis
        
    #save back into dataframe
    exponential_filtered_positions = pd.DataFrame(data=filtered_points, columns=['x', 'y', 'z'])
    exponential_filtered_positions.to_csv("exponential_filtered_positions.csv")
    return exponential_filtered_positions

positions = pd.read_csv('positions.csv')
inter_frame_correlation(positions)
