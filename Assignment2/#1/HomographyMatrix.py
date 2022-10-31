import cv2
import numpy as np
import matplotlib.pyplot as plt

src_points = np.array([[1335,978],[1333,1928],[2989,1950],[2988,952]])
dest_points = np.array([[299,856],[329,1948],[2131,1878],[2122,896]])

h, status = cv2.findHomography(src_points, dest_points)

im_src = cv2.imread('Amani_capture_isp_1.png')
im_dst = cv2.imread('Amani_capture_isp_2.png')

im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
print(h)
cv2.imshow("Warped_Source_Image", im_out)
plt.imshow(im_out)
plt.show()
#HM
#[[ 1.55385410e+00  4.56897321e-02 -1.76721065e+03]
#[ 1.58042282e-01  1.38417516e+00 -5.57351990e+02]
# [ 1.20560432e-04  1.62286723e-05  1.00000000e+00]]


