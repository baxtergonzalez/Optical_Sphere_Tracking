import cv2
import numpy as np
import yaml
import json
import pandas as pd
import time


# Load the camera calibration parameters
with open('baxter_webcam_calibration_matrix.yaml') as f:
    loadeddict = yaml.load(f, Loader=yaml.FullLoader)
    camera_matrix = np.array(loadeddict.get('camera_matrix'))
    dist_coeffs = np.array(loadeddict.get('dist_coeff'))
    rvecs = np.array(loadeddict.get('rvecs'))
    tvecs = np.array(loadeddict.get('tvecs'))
    focal_length_mm = loadeddict.get('focal_length_mm')

#Set some known parameters
SPHERE_DIAMETER = 4.0 #cm
FOCAL_LENGTH_PX = camera_matrix[0,0] #pixels
FOCAL_LENGTH_MM = focal_length_mm #mm

MIN_Z = 10.0 #cm
MAX_Z = 100.0 #cm

MAX_RAD = int((SPHERE_DIAMETER/2)*FOCAL_LENGTH_MM/MIN_Z)
MIN_RAD = int((SPHERE_DIAMETER/2)*FOCAL_LENGTH_MM/MAX_Z)

#Load color ranges
with open('color_ranges.json') as f:
    color_ranges = json.load(f)

positions = pd.DataFrame(columns=['x', 'y', 'z', 'color', 'frame_number', 'delta_time'])

def undistort_img(img: np.ndarray) -> np.ndarray:
    """Undistort the image using the camera calibration parameters."""
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, camera_matrix)
    return dst

def find_circles(img, color_name, target_radius, lower_color, upper_color, min_radius = MIN_RAD, max_radius = MAX_RAD):
    copy = img.copy()
    #Convert to HSV and pply blue color mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)

    #Apply mask to original img
    img = cv2.bitwise_and(img, img, mask=mask)

    #Morphological operations to improve mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Blur the image
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    #Find edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    #Morphological operations to improve edges
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)
    
    #draw edges on img
    img = cv2.addWeighted(copy, 0.5, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)

    #find circles
    try:
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        validated_circles = []
        for cnt in contours:
            base_area = cv2.contourArea(cnt)
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            circle_area = np.pi * radius**2

            #Check if the area of the contour is within 20 percent of the area of the circle binding it (To ensure that the contour is actually a circle)
            if abs(base_area - circle_area) > (circle_area * 0.2):
                # print("Not a circle")
                pass
            else:
                # print("Circle")
                cv2.circle(img, center, radius, (0, 255, 0), 2)
                validated_circles.append([center, radius, circle_area])
        if target_radius is None:
            #Find the largest circle
            largest_circle = max(validated_circles, key=lambda x: x[2])
            if largest_circle[1] < min_radius or largest_circle[1] > max_radius:
                # print("Circle too small or too large")
                return img, False, None, None
            else:
                cv2.circle(img, largest_circle[0], largest_circle[1], (0, 0, 255), 2)
                cv2.putText(img, color_name, largest_circle[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return img, True, largest_circle[0], largest_circle[1]
        else:
            #Find the circle closest to the target radius
            closest_circle = min(validated_circles, key=lambda x: abs(x[1] - target_radius))
            if closest_circle[1] < min_radius or closest_circle[1] > max_radius:
                # print("Circle too small or too large")
                return img, False, None, None
            else:
                if abs(closest_circle[1] - target_radius) > 1:
                    print(f'\n\n\nbig difference between target radius and closest circle radius: {abs(closest_circle[1] - target_radius)}')
                cv2.circle(img, closest_circle[0], closest_circle[1], (0, 0, 255), 2)
                cv2.putText(img, color_name, closest_circle[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                return img, True, closest_circle[0], closest_circle[1]
          
    except:
        # print("No circles found")
        return img, False, None, None
    
def estimate_sphere_pose(center, radius, prev_pose, frame_number, prev_frame_number, alpha=0.2):
    """Estimate the pose of the sphere in the camera frame."""
    print("Center: ", center)
    print("Radius: ", radius)

    z = (SPHERE_DIAMETER/2) * FOCAL_LENGTH_MM / radius
    x = -((center[0] - (camera_matrix[0,2])) * z / FOCAL_LENGTH_PX)
    y = -((center[1] - (camera_matrix[1,2])) * z / FOCAL_LENGTH_PX)

    pose = np.array([x, y, z])
    if prev_pose is not None and (frame_number - prev_frame_number < 10):
        # Apply exponential smoothing
        pose = alpha * pose + (1 - alpha) * prev_pose
        return pose
    else:
        return pose


# Load the image
cap = cv2.VideoCapture(0)
frame_number = 0
prev_time = time.time()
target_radius = None
prev_pose = None
prev_frame_number = 1

while True:
    frame_number += 1
    ret, frame = cap.read()
    img = undistort_img(frame)
    

    img, success, center_blue, radius_blue = find_circles(img, "blue", target_radius, np.array(color_ranges['blue']['lower']), np.array(color_ranges['blue']['upper']))
    # img, success, center_green, radius_green = find_circles(img, "green", np.array(color_ranges['green']['lower']), np.array(color_ranges['green']['upper']))
    if success:
        delta_time = time.time() - prev_time
        target_radius = radius_blue

        pose = estimate_sphere_pose(center_blue, radius_blue, prev_pose, frame_number, prev_frame_number, alpha=0.2)
        positions = pd.concat(objs=[positions, pd.DataFrame(data=[[pose[0], pose[1], pose[2], 'blue', frame_number, delta_time]], columns=['x', 'y', 'z', 'color', 'frame_number', 'delta_time'])], ignore_index=True)
        
        prev_frame_number = frame_number
        prev_pose = pose
        prev_time = time.time()
        print(f'Sphere pose: {pose}')

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

positions_csv = positions.to_csv('positions.csv', index=False)
print("saved positions.csv")

import process_data
import plot_positions