import face_recognition
import cv2
import numpy as np
import platform
import sqlite3
import pickle
import datetime

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []
known_face_metadata = []


# noinspection SqlResolve
class Jetson:

    def __init__(self):
        self.conn = None
        self.cursor = None

    def save_known_faces(self):
        with open("known_faces.dat", "wb") as face_data_file:
            face_data = [known_face_encodings, known_face_metadata]
            pickle.dump(face_data, face_data_file)
            print("Known faces backed up to disk.")

    def register_new_face(self, face_encoding, face_image):
        """
        Add a new person to our list of known faces
        """
        # Add the face encoding to the list of known faces
        known_face_encodings.append(face_encoding)
        # Add a matching dictionary entry to our metadata list.
        # We can use this to keep track of how many times a person has visited, when we last saw them, etc.
        known_face_metadata.append({
            "face_image": face_image,
            "person": "Persona {}".format(str(len(known_face_metadata)+1))
        })

    def load_known_faces(self):
        global known_face_encodings, known_face_metadata

        try:
            with open("known_faces.dat", "rb") as face_data_file:
                known_face_encodings, known_face_metadata = pickle.load(face_data_file)
                print("Known faces loaded from disk.")
        except FileNotFoundError as e:
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
        self.load_known_faces()
        # Get access to the webcam. The method is different depending on if this is running on a laptop or a Jetson Nano.
        if self.running_on_jetson_nano():
            # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
            video_capture = cv2.VideoCapture(self.get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
        else:
            # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
            # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
            video_capture = cv2.VideoCapture(0)


        while video_capture.isOpened():
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            i = 0
            for face_location, face_encoding in zip(face_locations, face_encodings):

                if len(known_face_encodings) != 0:
                    # Calculate the face distance between the unknown face and every face on in our known face list
                    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
                    # the more similar that face was to the unknown face.
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

                    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
                    best_match_index = np.argmin(face_distances)

                    # If the face with the lowest distance had a distance under 0.6, we consider it a face match.
                    # 0.6 comes from how the face recognition model was trained. It was trained to make sure pictures
                    # of the same person always were less than 0.6 away from each other.
                    # Here, we are loosening the threshold a little bit to 0.65 because it is unlikely that two very similar
                    # people will come up to the door at the same time.
                    if face_distances[best_match_index] < 0.65:
                        metadata = known_face_metadata[best_match_index]
                        print(metadata);
                        print("Persona {} su hash es {}".format(i, metadata["person"]))
                        # Draw a label with a name below the face
                        # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                        # font = cv2.FONT_HERSHEY_DUPLEX
                        # cv2.putText(frame, str(val), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                else:
                    # Add the new face to our known face data
                    top, right, bottom, left = face_location
                    face_image = small_frame[top:bottom, left:right]
                    face_image = cv2.resize(face_image, (150, 150))
                    self.register_new_face(face_encoding, face_image)
            # Display the final frame of video with boxes drawn around each detected fames
            # cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.save_known_faces()
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


d = Jetson()
d.principal()
