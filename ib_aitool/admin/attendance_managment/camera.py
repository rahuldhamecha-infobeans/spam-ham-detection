import os.path

from ib_tool import app
import cv2
from ib_aitool.admin.attendance_managment.attendance_model_train import predict
class Camera:
    def __init__(self,type='entry_camera'):
        self.type = type
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            app.logger.info('Cannot Open Camera')
            print("Cannot open camera")
            exit()

    def generated_frames(self):
        camera = self.camera
        process_this_frame = 29
        if camera is not None:
            while True:
                success, frame = camera.read()
                if not success:
                    return None
                else:

                    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    process_this_frame = process_this_frame + 1
                    model_path = os.path.join(app.config['UPLOAD_FOLDER'],'models','attendance_model.h5')
                    if process_this_frame % 30 == 0:
                        predictions = predict(img, model_path=model_path)
                        if predictions:
                            for name,(top,right,bottom,left) in predictions:
                                name = name.replace('_',' ')
                                name = name.upper()
                                cv2.putText(frame, name, (30, 60),
                                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    return frame

    def release(self):
        self.camera.release()
