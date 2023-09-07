from flask import render_template, Response, jsonify, render_template, Blueprint
from flask_restful import Api
from poc.objectdetection.camera import *


object_detection_blueprint = Blueprint(
    'objectdetection', __name__, template_folder='templates/')
api = Api(object_detection_blueprint)


@object_detection_blueprint.route('/')
def object_detection():
    navbar_brand = "<span>Live </span> Object Detection"
    return render_template('object-detection.html', navbar_brand=navbar_brand)


@object_detection_blueprint.route('/video-feed-for-object')
def video_feed_for_object():
    return Response(generate_webcam(VideoCameraForObject()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_webcam(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
