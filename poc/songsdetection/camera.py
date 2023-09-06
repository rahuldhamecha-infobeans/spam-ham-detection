import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pandastable import Table, TableModel
from tensorflow.keras.preprocessing import image
import datetime
from threading import Thread
# from Spotipy import *
import time
import pandas as pd
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
ds_factor = 0.6

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(
    3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights(os.path.join(os.path.dirname(
    __file__), 'models', 'model.h5'))


cv2.ocl.setUseOpenCL(False)


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

music_dist = {
    0: os.path.join(os.path.dirname(__file__), "songs", "angry.csv"),
    1: os.path.join(os.path.dirname(__file__), "songs", "disgusted.csv"),
    2: os.path.join(os.path.dirname(__file__), "songs", "fearful.csv"),
    3: os.path.join(os.path.dirname(__file__), "songs", "happy.csv"),
    4: os.path.join(os.path.dirname(__file__), "songs", "neutral.csv"),
    5: os.path.join(os.path.dirname(__file__), "songs", "sad.csv"),
    6: os.path.join(os.path.dirname(__file__), "songs", "surprised.csv"),
}


global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]

# import object names file
app_dir = os.path.dirname(os.path.abspath(__file__))
objectNames = os.path.join(app_dir, 'coco.names')
with open(objectNames, 'rt') as f:
    objectNames = f.read().rstrip('\n').split('\n')

''' Class for calculating FPS while streaming. Used this to check performance of using another thread for video streaming '''


''' Class for using another thread for video streaming to boost performance '''


''' Class for reading video stream, generating prediction and recommendations '''


class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

    def get_frame(self):
        global df1
        ret, image = self.cap.read()

        if not ret:
            print("Error reading frame from camera")
            return None, None

        if image is None:
            print("Empty frame received from camera")
            main.after(10, web_cam)  # Retry after a delay
            return None, None

        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

        df1 = pd.read_csv(music_dist[show_text[0]])
        df1 = df1[['Name', 'Album', 'Artist']]
        df1 = df1.head(15)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y - 50),
                          (x + w, y + h + 10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)

            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex
            cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            df1 = music_rec()

        global last_frame1
        last_frame1 = image.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(last_frame1)
        img = np.array(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), df1


def music_rec():
    # print('---------------- Value ------------', music_dist[show_text[0]])
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist']]
    df = df.head(15)
    return df

# Create video camera for live object detection.


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
