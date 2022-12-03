# run this code in YOLOv3-Object-Detection-with-OpenCV
import numpy as np
import argparse
import cv2 as cv
import subprocess
from yolo_utils import infer_image, show_image
import utils
import depthai as dai

def draw_text(img, text,
          font=cv.FONT_HERSHEY_PLAIN,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(0, 255, 0),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

scale = 3
wP = 210 *scale
hP= 297 *scale
width = 800
height = 600
pipeline = dai.Pipeline()
camRgb = pipeline.createColorCamera()
# cam.initialControl.setManualFocus(130)
xoutVideo = pipeline.createXLinkOut()
xoutVideo.setStreamName("video")
xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Properties
camRgb.setPreviewSize(width, height)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking to preview stream
camRgb.preview.link(xoutVideo.input)

FLAGS = []

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-m', '--model-path',
		type=str,
		default='./yolov3-coco/',
		help='The directory where the model weights and \
			  configuration files are.')

	parser.add_argument('-w', '--weights',
		type=str,
		default='./yolov3-coco/yolov3.weights',
		help='Path to the file which contains the weights \
			 	for YOLOv3.')

	parser.add_argument('-cfg', '--config',
		type=str,
		default='./yolov3-coco/yolov3.cfg',
		help='Path to the configuration file for the YOLOv3 model.')

	parser.add_argument('-i', '--image-path',
		type=str,
		help='The path to the image file')

	parser.add_argument('-v', '--video-path',
		type=str,
		help='The path to the video file')


	parser.add_argument('-vo', '--video-output-path',
		type=str,
        default='./output.avi',
		help='The path of the output video file')

	parser.add_argument('-l', '--labels',
		type=str,
		default='./yolov3-coco/coco-labels',
		help='Path to the file having the \
					labels in a new-line seperated way.')

	parser.add_argument('-c', '--confidence',
		type=float,
		default=0.5,
		help='The model will reject boundaries which has a \
				probabiity less than the confidence value. \
				default: 0.5')

	parser.add_argument('-th', '--threshold',
		type=float,
		default=0.3,
		help='The threshold to use when applying the \
				Non-Max Suppresion')

	parser.add_argument('--download-model',
		type=bool,
		default=False,
		help='Set to True, if the model weights and configurations \
				are not present on your local machine.')

	parser.add_argument('-t', '--show-time',
		type=bool,
		default=False,
		help='Show the time taken to infer each image.')

	FLAGS, unparsed = parser.parse_known_args()

	# Download the YOLOv3 models if needed
	if FLAGS.download_model:
		subprocess.call(['./yolov3-coco/get_model.sh'])

	# Get the labels
	labels = open(FLAGS.labels).read().strip().split('\n')

	# Intializing colors to represent each label uniquely
	colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

	# Load the weights and configutation to form the pretrained YOLOv3 model
	net = cv.dnn.readNetFromDarknet(FLAGS.config, FLAGS.weights)

	# Get the output layer names of the model
	layer_names = net.getLayerNames()
	print('names: ', layer_names)
	layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
	#outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
	# If both image and video files are given then raise error
	count = 0

	vid = cv.VideoCapture(0)
	with dai.Device(pipeline) as device:
		video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

		while True:
			videoIn = video.get()
			# Get BGR frame from NV12 encoded video frame to show with opencv
			frame = videoIn.getCvFrame()
			dimensions_frame = videoIn.getCvFrame()
			height, width = frame.shape[:2]

			if count == 0:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
																		height, width, frame, colors, labels, FLAGS)
				count += 1
			else:
				frame, boxes, confidences, classids, idxs = infer_image(net, layer_names, \
																		height, width, frame, colors, labels, FLAGS,
																		boxes, confidences, classids, idxs, infer=False)
				count = (count + 1) % 6

			#cv.imshow('oakD', frame)
			imgContours, conts = utils.getContours(dimensions_frame, minArea=50000, filter=4)
			#print('Img countours ', imgContours)

			#img = frame
			if len(conts) != 0:
				biggest = conts[0][2]
				print('biggest: ', biggest)
				imgWarp = utils.warpImg(dimensions_frame, biggest, wP, hP)
				imgContours2, conts2 = utils.getContours(imgWarp,
														 minArea=2000, filter=4,
														 cThr=[50, 50], draw=False)

				if len(conts) != 0:
					for obj in conts2:
						cv.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)

						nPoints = utils.reorder(obj[2])
						nW = round((utils.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
						nH = round((utils.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
						cv.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
										(nPoints[1][0][0], nPoints[1][0][1]),
										(255, 0, 255), 3, 8, 0, 0.05)
						print('Coordinates: ', (nPoints[0][0][0], nPoints[0][0][1]))
						cv.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
										(nPoints[2][0][0], nPoints[2][0][1]),
										(255, 0, 255), 3, 8, 0, 0.05)
						x, y, w, h = obj[3]
						#cv
						#cv.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL,
								#	1.5,
								#	(0, 0, 255), 2)
						draw_text(imgContours2, '{}cm'.format(nW), font_scale=4, pos=(x + 30, y - 10),
								  text_color_bg=(255, 0, 0))
						print('{}cm'.format(nW))
						print('{}cm'.format(nH))
						#cv.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2),
									#cv.FONT_HERSHEY_COMPLEX_SMALL,
								#	1.5,
								#	(0, 0, 255), 2)
						draw_text(imgContours2, '{}cm'.format(nH), font_scale=4, pos=(x - 70, y + h // 2),
								  text_color_bg=(255, 0, 0))

					# cv2.namedWindow("measurement", cv2.WND_PROP_FULLSCREEN)
					# cv2.setWindowProperty("measurement", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
					# print('imgContours2', imgContours2)
					#m = cv.resize(imgContours2, (0, 0), None, 3, 3)
				cv.imshow('measurement', imgContours2)
			#for contour in imgContours:
				#cv.drawContours(frame, contour, -1, (0, 255, 0), 3)
				#cv.drawContours(frame, [contour], 0, (0, 0, 255), 2)
			cv.imshow('Original', frame)
			if cv.waitKey(1) == ord('q'):
				break



	vid.release()
	cv.destroyAllWindows()
