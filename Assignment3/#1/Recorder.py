import cv2
import depthai as dai
# Create an object to read
# from cameravideo = cv2.VideoCapture(0)

# We need to check if camera
# is opened previously or not
pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("recorder")

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(860, 720)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

camRgb.video.link(xoutVideo.input)
# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'filename.avi' file.

with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="recorder", maxSize=1, blocking=False)
    #frame_width = int(video.get(3))
   # frame_height = int(video.get(4))

    #size = (frame_width, frame_height)
    #result = cv2.VideoWriter('video.mp4',
                            # cv2.VideoWriter_fourcc(*'fmp4'),
                            # 10, (640,480))
    #frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  #  frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = int(video.get(cv2.CAP_PROP_FPS))
    #print(frame_width, fps)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 10, (860, 720))
    while (True):
        videoIn = video.get()
        frame = videoIn.getCvFrame()
        #frame = frame.read()

        #if ret == True:

            # Write the frame into the
            # file 'filename.avi'
        out.write(frame)

            # Display the frame
            # saved in the file
        cv2.imshow('Frame', frame)

            # Press S on keyboard
            # to stop the process
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

        # Break the loop

    # When everything done, release
    # the video capture and video
    # write objects
    video.release()
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved")
