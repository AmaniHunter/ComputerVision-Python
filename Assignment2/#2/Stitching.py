import numpy as np
import cv2

imagesPaths1 = ["./GSU_Buildings/Set2/rialto1.jpeg","./GSU_Buildings/Set2/rialto2.jpeg","./GSU_Buildings/Set2/rialto3.jpeg"]
imagesPaths2 = ["./GSU_Buildings/Set1/studentcentereast1.jpg","./GSU_Buildings/Set1/studentcentereast2.jpg","./GSU_Buildings/Set1/studentcentereast3.jpg"]
imagesPaths3 = ["./GSU_Buildings/Set3/bookstore1.jpg","./GSU_Buildings/Set3/bookstore2.jpg","./GSU_Buildings/Set3/bookstore3.jpg"]
imagesPaths4 = ["./GSU_Buildings/Set4/bookstore5.jpg","./GSU_Buildings/Set4/bookstore6.jpg","./GSU_Buildings/Set4/bookstore7.jpg"]
imagesPaths5 = ["./GSU_Buildings/Set5/tdeck1.jpg","./GSU_Buildings/Set5/tdeck2.jpg","./GSU_Buildings/Set5/tdeck3.jpg"]
images = []

for path in imagesPaths2:
	image = cv2.imread(path)
	images.append(image)

stitcher = cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
print(stitched)
if status == 0:
	cv2.imshow("Stitched images", stitched)
	cv2.waitKey(0)
else:
	print("Failed to strich")

