import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False)

def getHomography(kps_train, kps_query, matches, reprojThresh):
    kpsA = np.float32([kp.pt for kp in kps_train])
    kpsB = np.float32([kp.pt for kp in kps_query])
    if len(matches) > 4:
        pointsA = np.float32([kpsA[m.queryIdx] for m in matches])
        pointsB = np.float32([kpsB[m.trainIdx] for m in matches])
        (Homography, status) = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, reprojThresh)
        print('matches: ', matches)
        return (matches, Homography, status)
    else:
        return None

def GetImageKeyPoints(image):
    descriptor = cv2.ORB_create()
    kps, features = descriptor.detectAndCompute(image, None)
    return (kps, features)

def MatchImageKeyPoints(features_train, features_query, ratio):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_match = bf.knnMatch(features_train, features_query, 2)
    matches = []

    for m, n in raw_match:
        if m.distance < n.distance * ratio:
            matches.append(m)

    return matches

def TransformImageToGrayScale(result):
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    result = result[y:y + h, x:x + w]
    return result


img_train = cv2.imread("../#1/Amani_capture_isp_1.png")
img_query = cv2.imread("../#1/Amani_capture_isp_2.png")
gray_img_train = cv2.cvtColor(img_train, cv2.COLOR_RGB2GRAY)
gray_img_query = cv2.cvtColor(img_query, cv2.COLOR_RGB2GRAY)
kps_train, features_train = GetImageKeyPoints(gray_img_train)
kps_query, features_query = GetImageKeyPoints(gray_img_query)
print(features_train,features_query, kps_train, kps_query)
matches = MatchImageKeyPoints(features_train, features_query, 0.75)
print(matches)
temp_img = cv2.drawMatches(img_train, kps_train, img_query, kps_query, np.random.choice(matches, 100), None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
M = getHomography(kps_train, kps_query, matches, 4)
matches, homography, status = M
width = img_train.shape[1] + img_query.shape[1]
height = img_train.shape[0] + img_query.shape[0]
imageResult = cv2.warpPerspective(img_train, homography, (width, height))
imageResult[0:img_query.shape[0], 0:img_query.shape[1]] = img_query
imageResult = TransformImageToGrayScale(imageResult)
plt.imshow(imageResult)
plt.show()

