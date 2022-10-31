import cv2
import matplotlib.pyplot as plt

captured_image = cv2.imread("Amani_capture_isp_1.png",0)
edges = cv2.Canny(captured_image, 180, 200)
plt.subplot(121),plt.imshow(captured_image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
