import cv2
import numpy as np

frame0 = cv2.imread('frame10.png')
frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
frame1 = cv2.imread('frame1.png')
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2 = cv2.imread('frame2.png')
frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
frame3 = cv2.imread('frame3.png')
frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
frame4 = cv2.imread('frame4.png')
frame4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
frame5 = cv2.imread('frame5.png')
frame5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2GRAY)
frame6 = cv2.imread('frame6.png')
frame6 = cv2.cvtColor(frame6, cv2.COLOR_BGR2GRAY)
frame7 = cv2.imread('frame7.png')
frame7 = cv2.cvtColor(frame7, cv2.COLOR_BGR2GRAY)
frame8 = cv2.imread('frame8.png')
frame8 = cv2.cvtColor(frame8, cv2.COLOR_BGR2GRAY)
frame9 = cv2.imread('frame9.png')
frame9 = cv2.cvtColor(frame9, cv2.COLOR_BGR2GRAY)

frame0 = cv2.resize(frame0, (87, 87))

frame1 = cv2.resize(frame1, (87, 87))

frame2 = cv2.resize(frame2, (87, 87))

frame3 = cv2.resize(frame3, (87, 87))

frame4 = cv2.resize(frame4, (87, 87))

frame5 = cv2.resize(frame5, (87, 87))

frame6 = cv2.resize(frame6, (87, 87))

frame7 = cv2.resize(frame7, (87, 87))

frame8 = cv2.resize(frame8, (87, 87))

frame9 = cv2.resize(frame9, (87, 87))

ssd0 = []

ssd01 = np.sum((np.square(frame0 - frame1)))

ssd0.append(ssd01)

ssd02 = np.sum((np.square(frame0 - frame2)))

ssd0.append(ssd02)

ssd03 = np.sum((np.square(frame0 - frame3)))

ssd0.append(ssd03)

ssd04 = np.sum((np.square(frame0 - frame4)))

ssd0.append(ssd04)

ssd05 = np.sum((np.square(frame0 - frame5)))

ssd0.append(ssd05)

ssd06 = np.sum((np.square(frame0 - frame6)))

ssd0.append(ssd06)

ssd07 = np.sum((np.square(frame0 - frame7)))

ssd0.append(ssd07)

ssd08 = np.sum((np.square(frame0 - frame8)))

ssd0.append(ssd08)

ssd09 = np.sum((np.square(frame0 - frame9)))

ssd0.append(ssd09)

print('SSD: ',ssd0)
corr0 = []

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

corr01 = correlation_coefficient(frame0, frame1)

corr0.append(corr01)

corr02 = correlation_coefficient(frame0, frame2)

corr0.append(corr02)

corr03 = correlation_coefficient(frame0, frame3)

corr0.append(corr03)

corr04 = correlation_coefficient(frame0, frame4)

corr0.append(corr04)

corr05 = correlation_coefficient(frame0, frame5)

corr0.append(corr05)

corr06 = correlation_coefficient(frame0, frame6)

corr0.append(corr06)

corr07 = correlation_coefficient(frame0, frame7)

corr0.append(corr07)

corr08 = correlation_coefficient(frame0, frame8)

corr0.append(corr08)

corr09 = correlation_coefficient(frame0, frame9)

corr0.append(corr09)

print('Correlation: ',corr0)