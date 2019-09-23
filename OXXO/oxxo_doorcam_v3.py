import face_recognition
import cv2
import numpy as np
import platform
import sqlite3
import pickle
import datetime
import requests
import copy
import json
import base64
import sys
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
import threading
import time
from playsound import playsound
import os


# noinspection SqlResolve
class Jetson:

    def __init__(self, argv):
        self.conn = None
        self.cursor = None
        # Our list of known face encodings and a matching list of metadata about each face.
        self.known_face_encodings = []
        self.known_face_metadata = []
        self.cache = []
        self.frame = None
        self.best_match = None
        self.loop = False
        if len(argv) == 1:
            self.accuracy = 0.60
        else:
            self.accuracy = float(argv[1])

    def getValue(self):
        return self.loop

    def setValue(self, condition):
        self.loop = condition

    def getAccuracy(self):
        return self.loop

    def setAccuracy(self, accuracy):
        self.accuracy = accuracy

    def save_known_faces(self):
        with open("known_faces.dat", "wb") as face_data_file:
            face_data = [self.known_face_encodings, self.known_face_metadata]
            pickle.dump(face_data, face_data_file)
            print("Known faces backed up to disk.")

    def register_new_face(self, face_encoding, face_image):
        """
        Add a new person to our list of known faces
        """
        # Add the face encoding to the list of known faces
        self.known_face_encodings.append(face_encoding)
        # Add a matching dictionary entry to our metadata list.
        # We can use this to keep track of how many times a person has visited, when we last saw them, etc.

        value = {
            "face_image": face_image,
            "person": "Persona {}".format(str(len(self.known_face_metadata)))
        }

        self.known_face_metadata.append(value)

        return value

    def load_face_metadata(self, face_enconding):
        # Calculate the face distance between the unknown face and every face on in our known face list
        # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
        # the more similar that face was to the unknown face.
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_enconding)

        # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
        best_match_index = np.argmin(face_distances)

        # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
        # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
        # of the same person always were less than 0.6 away from each other.
        # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
        # people will come up to the door at the same time.

        print(face_distances[best_match_index])
        if face_distances[best_match_index] < self.accuracy:
            return self.known_face_metadata[best_match_index]

        return None

    def load_known_faces(self):

        try:
            with open("known_faces.dat", "rb") as face_data_file:
                self.known_face_encodings, self.known_face_metadata = pickle.load(face_data_file)
                print("Known faces loaded from disk.")
        except FileNotFoundError:
            print("No previous face data found - starting with a blank known face list.")
            pass

    def activation(self, x):
        return np.sinh(x) / np.cosh(x)

    def load_faces(self, json):
        try:
            self.conn = sqlite3.connect(":memory:")
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(e)

        self.cursor.execute("CREATE TABLE if NOT EXISTS faces (id integer, hash text, name text)")

        i = 0
        if len(json) > 0:

            while i < len(json):
                self.cursor.execute(
                    "INSERT INTO faces VALUES(" + str(i) + ",'" + str(json[i]["hash"]) + "','" + json[i]["name"] + "')")
                i = i + 1
            self.conn.commit()

        self.cursor.execute("SELECT * FROM faces")

        return self.cursor.fetchall()

    def running_on_jetson_nano(self):
        # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
        # so that we can access the camera correctly in that case.
        # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
        return platform.machine() == "aarch64"

    def get_jetson_gstreamer_source(self, capture_width=640, capture_height=480, display_width=640, display_height=480,
                                    framerate=60, flip_method=0):
        """
        Return an OpenCV-compatible video source description that uses gstreamer to capture video from the camera on a Jetson Nano
        """
        return (
                f'nvarguscamerasrc ! video/x-raw(memory:NVMM), ' +
                f'width=(int){capture_width}, height=(int){capture_height}, ' +
                f'format=(string)NV12, framerate=(fraction){framerate}/1 ! ' +
                f'nvvidconv flip-method={flip_method} ! ' +
                f'video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! ' +
                'videoconvert ! video/x-raw, format=(string)BGR ! appsink'
        )

    def principal(self):
        print("Start Call")
        # self.load_known_faces()
        # Get access to the webcam. The method is different depending on if this is running on a laptop or a Jetson Nano.
        if self.running_on_jetson_nano():
            # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
            video_capture = cv2.VideoCapture(self.get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
        else:
            # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
            # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
            video_capture = cv2.VideoCapture(0)

        while video_capture.isOpened() and self.loop:
            # Grab a single frame of video
            ret, self.frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(self.frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            current = []
            print("Detecto {} caras en {}".format(len(face_locations), str(datetime.datetime.utcnow())))
            for face_location, face_encoding in zip(face_locations, face_encodings):

                if len(self.known_face_encodings) != 0:

                    self.best_match = self.load_face_metadata(face_encoding)

                    if self.best_match is not None:
                        print("Hash es {}".format(self.best_match["person"]))
                        # Draw a label with a name below the face
                        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        # font = cv2.FONT_HERSHEY_DUPLEX
                        # cv2.putText(frame, str(val), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                    else:
                        print("Registrando una nueva cara")
                        # Add the new face to our known face data
                        top, right, bottom, left = face_location
                        face_image = small_frame[top:bottom, left:right]
                        face_image = cv2.resize(face_image, (150, 150))
                        self.best_match = self.register_new_face(face_encoding, face_image)
                else:
                    print("Registrando una nueva cara")
                    # Add the new face to our known face data
                    top, right, bottom, left = face_location
                    face_image = small_frame[top:bottom, left:right]
                    face_image = cv2.resize(face_image, (150, 150))
                    self.best_match = self.register_new_face(face_encoding, face_image)

                current.append({"face_encoding": face_encoding, "best_match": copy.deepcopy(self.best_match),
                                "face_location": face_location, "timestamp": str(datetime.datetime.utcnow())})

            send = False
            if len(current) != len(self.cache):
                print("Lo envio")
                send = True
            else:
                for data in current:
                    face_distances = face_recognition.face_distance(self.cache, data.get("face_encoding"))
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] > 0.65:
                        print("Lo envio")
                        send = True
                        break

            if send:
                print("Lo enviando")
                self.cache = list(map(self.data_parse, current))
                response = list(map(self.generate_request, current))

                headers = {'Content-type': 'application/json'}
                if len(response) > 0:
                    requests.post("http://localhost:3000/sendAll",
                                  data=json.dumps({"payload": response}), headers=headers)

            # cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.save_known_faces()
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def data_parse(self, datos):
        return datos.get("face_encoding")

    def generate_request(self, dato):

        height, width, _ = self.frame.shape

        top, right, bottom, left = dato.get("face_location")

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        location = self.getCoordinates(width, height,
                                       {"top": top, "left": left, "height": bottom - top, "width": right - left})

        # Copy image no reference
        img = copy.deepcopy(self.frame)

        crop_img = img[location["top"]:location["top"] + location["height"],
                   location["left"]:location["left"] + location["width"]]

        buffer = cv2.imencode('.jpg', crop_img)
        jpg_as_text = base64.b64encode(buffer[1]).decode()

        return {"identified": True, "name": dato.get("best_match")["person"], "timestamp": dato.get("timestamp"),
                "image": "data:image/jpeg;base64," + jpg_as_text}

    def getCoordinates(self, width, height, face):

        initial = {
            "w": width if face["width"] + 80 > width else face["width"] + 80,
            "h": height if face["height"] + 80 > height else face["height"] + 80,
            "l": 0 if face["left"] - 40 < 0 else face["left"] - 40,
            "t": 0 if face["top"] - 40 < 0 else face["top"] - 40,
        }

        delta = {
            "w": initial["w"] - width if initial["w"] > width else 0,
            "h": initial["h"] - height if initial["h"] > height else 0,
            "l": 0 - initial["l"] if initial["l"] < 0 else 0,
            "t": 0 - initial["t"] if initial["t"] < 0 else 0,
        }

        final = {
            "width": 0,
            "height": 0,
            "left": 0,
            "top": 0
        }

        if delta["w"] == 0 and 0 == delta["l"]:
            final["width"] = initial["w"]
            final["left"] = initial["l"]
        elif delta["w"] > delta["l"]:
            final["width"] = width
            if delta["l"] == 0:
                final["left"] = initial["l"] + delta["w"]
            else:
                final["left"] = initial["l"] + delta["w"] - delta["l"]
        else:
            final["left"] = 0
            if delta["w"] == 0:
                final["width"] = initial["w"] - delta["l"]
            else:
                final["width"] = initial["w"] - delta["l"] + delta["w"]

        if delta["h"] == 0 and 0 == delta["t"]:
            final["height"] = initial["h"]
            final["top"] = initial["t"]
        elif delta["h"] > delta["t"]:
            final["height"] = height
            if delta["t"] == 0:
                final["top"] = initial["t"] + delta["h"]
            else:
                final["top"] = initial["t"] + delta["h"] - delta["t"]
        else:
            final["top"] = 0
            if delta["h"] == 0:
                final["height"] = initial["h"] - delta["t"]
            else:
                final["height"] = initial["h"] - delta["t"] + delta["h"]

        if final["width"] + final["left"] > width:
            delta = (final["width"] + final["left"]) - width
            final["left"] = 0 if final["left"] - delta < 0 else final["left"] - delta

        if final["height"] + final["top"] > height:
            delta = (final["height"] + final["top"]) - height
            final["top"] = 0 if final["top"] - delta < 0 else final["top"] - delta

        return final


d = Jetson(sys.argv)

app = Flask(__name__, template_folder="../templates")
threader = threading.Event()


@app.route("/start")
def hello():
    if not d.getValue():
        d.setValue(True)
        threading.Timer(3, d.principal, []).start()
        message = "Server Started"
        return jsonify(
            status=200,
            message=message
        )
    else:
        message = "Server is running"
        return jsonify(
            status=201,
            message=message
        )


@app.route("/stop", methods=["post"])
def stop():
    d.setValue(False)
    message = "Server Stop"
    return jsonify(
        status=200,
        message=message
    )


@app.route("/restart", methods=["post"])
def restart():
    stop()
    time.sleep(2)
    return hello()


@app.route("/")
def start():
    return hello()


@app.route("/sound", methods=["get"])
def sound():
    print(os.getcwd() + '/beep.wav')
    playsound(os.getcwd() + '/beep.wav')
    return jsonify(
        status=200,
        message=os.getcwd() + '/beep.wav'
    )


@app.route("/accuracy", methods=["post"])
def accuracy():
    d.setAccuracy(request.form['accuracy'])
    return restart()


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
