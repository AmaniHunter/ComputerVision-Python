import numpy as np
import cv2 as cv
from scipy.spatial.transform import Rotation
from math import cos, sin, radians
import num2help as num2

rvecs = num2.rvecs
tvecs = num2.tvecs
r_obj = Rotation.from_rotvec(np.array(rvecs[0]).reshape(1,3))
rot_matrix = r_obj.as_matrix()
nprvecs = np.array(rvecs)
nptvecs = np.array(tvecs)

xc, xs = cos(radians(nprvecs[0][0][0])), sin(radians(nprvecs[0][0][0]))
yc, ys = cos(radians(nprvecs[0][1][0])), sin(radians(nprvecs[0][1][0]))
zc, zs = cos(radians(nprvecs[0][2][0])), sin(radians(nprvecs[0][2][0]))

tx = nptvecs[0][0][0]
ty = nptvecs[0][1][0]
tz = nptvecs[0][2][0]
translation_mtx = np.array([[1,0,0,tx],[0,1,0,ty],[0,0,1,tz],[0,0,0,1]])

rotation_x_matrix = np.array([[1, 0, 0, 0], [0, xc, -xs, 0], [0, xs, -xc, 0], [0, 0, 0, 1]])
rotation_y_matrix = np.array([[yc, 0, ys, 0], [0, 1, 0, 0], [-ys, 0, yc, 0], [0, 0, 0, 1]])
rotation_z_matrix = np.array([[zc, -zs, 0, 0], [zs, zc, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
extrinsinc_matrix = np.dot(rotation_z_matrix, np.dot(rotation_y_matrix, np.dot(rotation_x_matrix, translation_mtx)))
intrinsic_matrix = np.append(np.append(num2.mtx, [[0], [0], [1]], axis=1), [np.array([0, 0, 0, 1])], axis=0)
print('Intrinsinc matrix: ', intrinsic_matrix)
print('Extrinsinc matrix, ', extrinsinc_matrix)

camera_matrix = np.dot(intrinsic_matrix, extrinsinc_matrix)
print('camera_matrix: ', camera_matrix)
cv.destroyAllWindows()

inverse_matrix = np.linalg.inv(camera_matrix)
project_points = np.array([[5],[10],[30],[1]])
real_dimensions = inverse_matrix.dot(project_points)

print("Dimensions")
print(real_dimensions)
print("X Axis Length: ", real_dimensions[0][0])
print("Y Axis Length: ", real_dimensions[1][0])
print("Z Axis Length: ", real_dimensions[2][0])
