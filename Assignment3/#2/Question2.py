import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt

def motion_constraints(Iref, Inext):
    kernel_x = np.array([[-1., 1.], [-1., 1.]])*.25
    kernel_y = np.array([[-1., -1.], [1., 1.]])*.25
    kernel_t = np.array([[1., 1.], [1., 1.]])*.25
    Iref = Iref / 255. # normalize pixels
    Inext = Inext / 255. # normalize pixels
    mode = 'same'

    Ix = signal.convolve2d(Iref, kernel_x, boundary='symm', mode=mode)
    Iy = signal.convolve2d(Iref, kernel_y, boundary='symm', mode=mode)
    It = signal.convolve2d(Inext, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(Iref, -kernel_t, boundary='symm', mode=mode)
    return Ix, Iy, It
frameA = "../#1/frame1.png"
frameB = "../#1/frame2.png"
frame0 = cv2.imread(frameA)
frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
frame0 = cv2.resize(frame0, (87,87))
frame1 = cv2.imread(frameB)
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1 = cv2.resize(frame1, (87,87))

Ix, Iy, It = motion_constraints(frame0, frame1)
print(Ix, Iy, It)