import cv2
import depthai as dai
import numpy as np
import face_recognition

pipeline = dai.Pipeline()

camRgb = pipeline.create(dai.node.ColorCamera)
xoutVideo = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("face detector")

camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setVideoSize(860, 720)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

camRgb.video.link(xoutVideo.input)

amani_image = face_recognition.load_image_file("Amani_pic.jpeg")
amani_face_encoding = face_recognition.face_encodings(amani_image)[0]

known_face_encodings = [
    amani_face_encoding,
]
known_face_names = [
    "Amani Hunter"
]
val = input("Enter name of person to detect: ")
print(val)

face_locations = []
face_encodings = []
face_names = []
top_lip = []
bottom_lip = []
center_points = []
process_this_frame = True

with dai.Device(pipeline) as device:
    video = device.getOutputQueue(name="face detector", maxSize=1, blocking=False)

    while True:
        videoIn = video.get()
        frame = videoIn.getCvFrame()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
            face_names = []
            for index, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown Individual"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                if name == 'Amani Hunter' and val == name:
                    keys = list(face_landmarks_list[index].keys())
                    top_lip = face_landmarks_list[index][keys[-2]]
                    bottom_lip = face_landmarks_list[index][keys[-1]]
                    top_lip = np.array(top_lip, dtype=np.int32)
                    bottom_lip = np.array(bottom_lip, dtype=np.int32)
                    top_lip = top_lip * 4
                    bottom_lip = bottom_lip * 4
                    center_top_lip = np.mean(top_lip, axis=0)
                    center_top_lip = center_top_lip.astype('int')
                    center_points.append(center_top_lip)
                face_names.append(name)
        process_this_frame = not process_this_frame

       # cv2.polylines(frame, np.array([top_lip]), 1, (255, 255, 255))
        #cv2.polylines(frame, np.array([bottom_lip]), 1, (255, 255, 255))
        for i in range(1, len(center_points)):
            if center_points[i - 1] is None or center_points[i] is None:
                continue
           # cv2.line(frame, tuple(center_points[i - 1]), tuple(center_points[i]), (0, 0, 255), 2)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Face Detector', frame)
        if cv2.waitKey(1) == ord('q'):
            break
cv2.destroyAllWindows()
