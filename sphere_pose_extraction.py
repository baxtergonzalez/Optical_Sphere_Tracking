import cv2
import numpy as np
import yaml
import json
import pandas as pd
import time


from matplotlib import pyplot as plt
import matplotlib.widgets

#Choose whether to plot the data or not
want_plot = input('Do you want to plot the data in real time? (y/n)')
if want_plot == 'y':
    PLOT = True
else:
    PLOT = False

# Load the camera calibration parameters (obtained from calibration procedure in opencv)
with open('baxter_webcam_calibration_matrix.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    camera_matrix = np.array(loadeddict.get('camera_matrix'))
    dist_coeffs = np.array(loadeddict.get('dist_coeff'))
    rvecs = np.array(loadeddict.get('rvecs'))
    tvecs = np.array(loadeddict.get('tvecs'))
    focal_length_mm = loadeddict.get('focal_length_mm')

#Load color ranges for masking spheres
with open('color_ranges.json') as f:
    color_ranges = json.load(f)


#Set some known parameters...
SPHERE_DIAMETER = 4.0 #cm
FOCAL_LENGTH_PX = camera_matrix[0,0] #pixels
FOCAL_LENGTH_MM = focal_length_mm #mm

#Distance range to look for spheres (in cm)
MIN_Z = 10.0
MAX_Z = 100.0

#Range of radii to look for spheres (in pixels)
MAX_RAD = int((SPHERE_DIAMETER/2)*FOCAL_LENGTH_MM/MIN_Z)
MIN_RAD = int((SPHERE_DIAMETER/2)*FOCAL_LENGTH_MM/MAX_Z)


if PLOT:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()


    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.scatter(0, 0, 0, c='r', marker='o')

def capture_frame(cap):
    """Capture a frame from the camera and undistort it.

    Args:
        cap (cv2.VideoCapture): Video capture object

    Returns:
        np.ndarray: Undistorted image from camera
    """
    ret, frame = cap.read()
    frame = undistort_img(frame)
    return frame

def undistort_img(img):
    """Undistort the image using the camera calibration parameters.

    Args:
        img (np.ndarray): Input image from camera

    Returns:
        np.ndarray: Image with camera distortion removed
    """
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    return dst

def apply_mask(img, lower_color, upper_color, kernel_size = 5):
    """Apply a color mask to the image.

    Args:
        img (np.ndarray): Input image
        lower_color (np.ndarray): Lower bound of color range
        upper_color (np.ndarray): Upper bound of color range

    Returns:
        np.ndarray: Image with color mask applied
    """
    lower_color = np.array(lower_color)
    upper_color = np.array(upper_color)

    #Convert to HSV and pply blue color mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    #Apply mask to original img
    img = cv2.bitwise_and(img, img, mask=mask)

    #Morphological operations to improve mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return img

def extract_edges(masked_image, kernel_size = 5):
    """Extract edges from the masked image.

    Args:
        masked_image (np.ndarray): Image with color mask applied

    Returns:
        np.ndarray: Image with edges extracted
    """
    #Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    #Blur the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #Find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #Morphological operations to improve edges
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    return edges

def find_circles(img, color_name, target_radius, min_radius = MIN_RAD, max_radius = MAX_RAD):
    """Look to find circles in the image based on the color mask

    Args:
        img (np.ndarray): Input image
        color_name (str): Name of the color to look for
        target_radius (int): Radius of the circle to look for (in pixels)
        min_radius (int, optional): Minimum radius to look for. Defaults to MIN_RAD.
        max_radius (int, optional): Maximum radius to look for. Defaults to MAX_RAD.

    """
    #Apply color mask and extract edges
    masked_image = apply_mask(img, color_ranges[color_name]['lower'], color_ranges[color_name]['upper'], kernel_size = 5)
    edges = extract_edges(masked_image)

    #Find circles in the image
    try:
        #Find contours in the image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        validated_circles = []
        #Loop through all contours found, and keep track of those that are valid circles...
        for cnt in contours:
            #Determine contour area, and area of bounding circle for each contour found
            base_area = cv2.contourArea(cnt)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            circle_area = np.pi * radius**2

            #Check if the area of the contour is within 20 percent of the area of the circle binding it (To ensure that the contour is actually a circle)
            if abs(base_area - circle_area) > (circle_area * 0.2):
                #Not a good enough match to be a circle
                pass
            else:
                #Validated circle!
                #Draw the circle on the original image in green, include it in list of validated circles
                cv2.circle(img, center, radius, (0, 255, 0), 2)
                validated_circles.append([center, radius, circle_area])
        
        #Assuming that previous frame was not a valid detection, simply return the largest circle
        if target_radius is None:
            #Find the largest circle
            largest_circle = max(validated_circles, key=lambda x: x[2])
            if largest_circle[1] < min_radius or largest_circle[1] > max_radius:
                #Circle not within range of acceptable radii, therefore not a valid detection
                return img, False, None, None
            else:
                #Draw the largest circle on the original image in red, add text to indicate color, and return it
                cv2.circle(img, largest_circle[0], largest_circle[1], (0, 0, 255), 2)
                cv2.putText(img, color_name, largest_circle[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return img, True, largest_circle[0], largest_circle[1]
        
        #Assuming that previous frame was a valid detection, find the circle closest to the target radius (To ensure that the circle is not moving too much)
        else:
            closest_circle = min(validated_circles, key=lambda x: abs(x[1] - target_radius))
            if closest_circle[1] < min_radius or closest_circle[1] > max_radius:
                #Circle too large or too small, therefore not a valid detection
                return img, False, None, None
            else:
                #Draw the closest circle on the original image in red, add text to indicate color, and return it
                if abs(closest_circle[1] - target_radius) > 1:
                    #Make a note to the user if the circle is moving too much
                    print(f'\n\n\nbig difference between target radius and closest circle radius: {abs(closest_circle[1] - target_radius)}')
                cv2.circle(img, closest_circle[0], closest_circle[1], (0, 0, 255), 2)
                cv2.putText(img, color_name, closest_circle[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return img, True, closest_circle[0], closest_circle[1]
    
    #In the case that no contours are found, return the original image, assuming that no circles were found  
    except:
        #No circles found, return original image
        return img, False, None, None

def estimate_sphere_pose(center, radius, prev_pose, frame_number, prev_frame_number, alpha = 0.2):
    """Estimate the pose of the sphere in the camera frame.
    
    Args:
        center (tuple): Center of the sphere in the image
        radius (int): Radius of the sphere in the image
        prev_pose (np.ndarray): Previous pose of the sphere
        frame_number (int): Current frame number
        prev_frame_number (int): Previous frame number where a valid pose was found
        alpha (float, optional): Exponential smoothing factor. Defaults to 0.2. (smaller is more smoothing)

    Returns:
        np.ndarray: Estimated pose of the sphere in the camera frame (x,y,z)
    
    """
    #Estimate depth of sphere based on radius in pixels, known radius of sphere in cm, and focal length of camera in mm
    #NOTE:
    z = (SPHERE_DIAMETER/2) * (FOCAL_LENGTH_MM)/ radius
    #Estimate x and y position of sphere based on center of sphere in pixels, and depth of sphere using similar triangles
    x = -((center[0] - (camera_matrix[0,2])) * z / FOCAL_LENGTH_PX)
    y = -((center[1] - (camera_matrix[1,2])) * z / FOCAL_LENGTH_PX)

    pose = np.array([x, y, z])
    if prev_pose is not None and (frame_number - prev_frame_number < 10):
        # Apply exponential smoothing
        pose = alpha * pose + (1 - alpha) * prev_pose
        return pose
    else:
        return pose

def update_plot(positions):
    if len(positions) == 0:
        pass
    else:
        ax.clear()

        ax.set_xlim3d(-50, 50)
        ax.set_ylim3d(-50, 50)
        ax.set_zlim3d(-50, 50)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for index, row in positions.iterrows():
            ax.scatter(row['x'], row['z'], row['y'], c='r', marker='o')
        plt.show(block = False)
    plt.pause(.001)

def main():
    # Load the image
    cap = cv2.VideoCapture(0)

    #Initialize variables
    frame_number = 0
    prev_time = time.time()
    target_radius = None
    prev_pose = None
    prev_frame_number = 1

    #Create a dataframe to store the sphere positions
    """
    x,y,z are the coordinates of the sphere in the camera frame
    color is the color of the sphere detected
    frame_number is the frame in which the sphere was detected
    delta_time is the time in seconds since the previous sphere was detected
    """
    positions = pd.DataFrame(columns=['x', 'y', 'z', 'color', 'frame_number', 'delta_time'])

    #Main loop
    while True:
        #Capture current frame, preprocess it
        frame_number += 1
        frame = capture_frame(cap)
        img = undistort_img(frame)

        #Detect the sphere of specified color
        img, success, center, radius = find_circles(img, "orange", target_radius)

        #If a valid detection is found, estimate the pose of the sphere
        if success:
            #Update change in time
            delta_time = time.time() - prev_time
            
            #Estimate the pose of the sphere, and add it to the positions dataframe
            pose = estimate_sphere_pose(center, radius, prev_pose, frame_number, prev_frame_number, alpha=0.2)
            positions = pd.concat(objs=[positions, pd.DataFrame(data=[[pose[0], pose[1], pose[2], 'orange', frame_number, delta_time]], columns=['x', 'y', 'z', 'color', 'frame_number', 'delta_time'])], ignore_index=True)
            
            #Update variables for next iteration
            target_radius = radius
            prev_frame_number = frame_number
            prev_pose = pose
            prev_time = time.time()
            print(f'Sphere pose: {pose}')
            
        if PLOT:
            update_plot(positions)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quitting video loop...')
            break

    #Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    #Save the positions dataframe to a csv file for plotting later
    positions.to_csv('positions.csv', index=False)
    print('Saved positions to positions.csv')



#Run the main function
main()
if not PLOT:
    import plot_positions