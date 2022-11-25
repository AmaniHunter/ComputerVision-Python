import cv2
import numpy as np
import numba as nb
import depthai as dai

streams = []
streams.append('isp')

@nb.njit(nb.uint16[::1] (nb.uint8[::1], nb.uint16[::1], nb.boolean), parallel=True, cache=True)
def unpack_raw10(input, out, expand16bit):
    lShift = 6 if expand16bit else 0

    for i in np.arange(input.size // 5):
        b4 = input[i * 5 + 4]
        out[i * 4]     = ((input[i * 5]     << 2) | ( b4       & 0x3)) << lShift
        out[i * 4 + 1] = ((input[i * 5 + 1] << 2) | ((b4 >> 2) & 0x3)) << lShift
        out[i * 4 + 2] = ((input[i * 5 + 2] << 2) | ((b4 >> 4) & 0x3)) << lShift
        out[i * 4 + 3] = ((input[i * 5 + 3] << 2) |  (b4 >> 6)       ) << lShift

    return out

print("depthai version:", dai.__version__)
pipeline = dai.Pipeline()

cam = pipeline.createColorCamera()
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)

if 'isp' in streams:
    xout_isp = pipeline.createXLinkOut()
    xout_isp.setStreamName('isp')
    cam.isp.link(xout_isp.input)

device = dai.Device(pipeline)
device.startPipeline()

q_list = []
for s in streams:
    q = device.getOutputQueue(name=s, maxSize=3, blocking=True)
    q_list.append(q)
    cv2.namedWindow(s, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(s, (960, 540))

capture_flag = False
image_counter = 0
while True:
    for q in q_list:
        name = q.getName()
        data = q.get()
        width, height = data.getWidth(), data.getHeight()
        payload = data.getData()
        capture_file_info_name = f"frame{image_counter}"
        if name == 'isp':
            shape = (height * 3 // 2, width)
            yuv420p = payload.reshape(shape).astype(np.uint8)
            bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)
            grayscale_img = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        if capture_flag:
            filename = capture_file_info_name + '.png'
            print(filename)
            grayscale_img = np.ascontiguousarray(grayscale_img)
            cv2.imwrite(filename, grayscale_img)
        bgr = np.ascontiguousarray(bgr)
        cv2.imshow(name, grayscale_img)
    capture_flag = False
    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    elif key%256 == 32:
        capture_flag = True
        image_counter += 1