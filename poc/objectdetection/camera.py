import os
import numpy as np
import cv2
# import object names file
app_dir = os.path.dirname(os.path.abspath(__file__))
objectNames = os.path.join(app_dir, 'coco.names')
with open(objectNames, 'rt') as f:
    objectNames = f.read().rstrip('\n').split('\n')


class VideoCameraForObject(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def get_frame(self):
        thres = 0.45  # Threshold to detect object
        configPath = os.path.join(
            app_dir, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
        weightsPath = os.path.join(app_dir, 'frozen_inference_graph.pb')

        net = cv2.dnn_DetectionModel(weightsPath, configPath)
        net.setInputSize(320, 320)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)

        success, img = self.cap.read()

        if not success:
            print("Error reading frame from camera")
            return None, None

        if img is None:
            print("Empty frame received from camera")
            main.after(10, web_cam)  # Retry after a delay
            return None, None

        if success:
            classIds, confs, bbox = net.detect(img, confThreshold=thres)
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                if classId - 1 < len(objectNames):
                    class_name = objectNames[classId - 1].upper()
                    cv2.rectangle(img, box, color=(179, 27, 0), thickness=1)
                    cv2.putText(img, class_name, (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()
