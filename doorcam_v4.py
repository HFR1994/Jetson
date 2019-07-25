import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
import re
import platform
import face_recognition
import requests
import json
import base64
import copy

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class Jetson:

    def __init__(self):
        # STEP 1: Train the KNN classifier and save it to disk
        # Once the model is trained and saved, you can skip this step next time.
        print("Training KNN classifier...")
        self.train("img-data/faces", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete!")

        frames = []
        frame_count = 0
        faces_in_batch = {}

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

            # Quit when the input video file ends
            if not ret:
                break

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Save each frame of the video to a list
            frame_count += 1
            frames.append(rgb_small_frame)
            batch_size = 32

            # Every 32 frames (the default batch size), batch process the list of frames to find faces
            if len(frames) == batch_size:

                print("Starting Batch")
                batch_of_face_locations = face_recognition.batch_face_locations(frames, batch_size=batch_size)

                print("Finished Batch")

                # Now let's list all the faces we found in all 128 frames
                for _, face_locations in enumerate(batch_of_face_locations):

                    number_of_faces_in_frame = len(face_locations)

                    if number_of_faces_in_frame != 0:
                        faces_encodings = face_recognition.face_encodings(rgb_small_frame,
                                                                          known_face_locations=face_locations)
                        predictions = self.predict(face_locations, faces_encodings, model_path="trained_knn_model.clf")
                        # Print results on the console
                        for name, (top, right, bottom, left) in predictions:
                            print("Found face: ", name)
                            # top *= 4
                            # right *= 4
                            # bottom *= 4
                            # left *= 4

                            height, width, _ = frame.shape

                            coor = self.getCoordinates(width, height, {"top": top, "left": left, "height": bottom - top,
                                                                       "width": right - left})

                            if name not in faces_in_batch:
                                faces_in_batch[name] = []

                            faces_in_batch[name].append({"frame": frame, "face_location": coor})

                response = {}

                for key, value in faces_in_batch.items():
                    if key != "unknown":
                        middle = int(len(value) / 2)

                        selected = value[middle]["frame"]
                        location = value[middle]["face_location"]

                        # Copy image no reference
                        img = copy.deepcopy(selected)

                        crop_img = img[location["top"]:location["top"] + location["height"],
                                   location["left"]:location["left"] + location["width"]]

                        buffer = cv2.imencode('.jpg', crop_img)
                        jpg_as_text = base64.b64encode(buffer[1]).decode()

                        response[key] = {"face_location": location, "image": "data:image/jpeg;base64," + jpg_as_text}

                frames = []
                frame_count = 0
                faces_in_batch = {}

                headers = {'Content-type': 'application/json'}
                while len(response) > 0:
                    requests.post("https://orquestrator-visual.mybluemix.net/facePaint", data=json.dumps({"payload": response}), headers=headers)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

    def running_on_jetson_nano(self):
        # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
        # so that we can access the camera correctly in that case.
        # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
        return platform.machine() == "aarch64"

    def get_jetson_gstreamer_source(self, capture_width=1280, capture_height=720, display_width=1280,
                                    display_height=720,
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

    def predict(self, faces_locations, faces_encondings, knn_clf=None, model_path=None, distance_threshold=0.6):

        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encondings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(faces_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encondings), faces_locations, are_matches)]

    def video_files_in_folder(self, folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(mp4)', f, flags=re.I)]

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

    def gdc(self, p, q):
        if q == 0:
            return p
        else:
            return self.gdc(q, p % q)

    def train(self, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        X = []
        y = []

        if model_save_path is not None and os.path.exists(model_save_path):
            with open(model_save_path, 'rb') as f:
                knn_clf = pickle.load(f)
                return knn_clf

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for video_path in self.video_files_in_folder(os.path.join(train_dir, class_dir)):
                input_movie = cv2.VideoCapture(video_path)

                frame_number = 0

                while True:
                    # Grab a single frame of video
                    ret, frame = input_movie.read()
                    frame_number += 1

                    # Quit when the input video file ends
                    if not ret:
                        break

                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_frame = frame[:, :, ::-1]

                    # Find all the faces and face encodings in the current frame of video
                    face_bounding_boxes = face_recognition.face_locations(rgb_frame)

                    if len(face_bounding_boxes) != 1:
                        # If there are no people (or too many people) in a training image, skip the image.
                        if verbose:
                            print("Image {} not suitable in frame {} for training: {}".format(video_path, str(frame),
                                                                                              "Didn't find a face" if len(
                                                                                                  face_bounding_boxes) < 1 else "Found more than one face"))
                    else:
                        # Add face encoding for current image to the training set
                        X.append(
                            face_recognition.face_encodings(rgb_frame, known_face_locations=face_bounding_boxes)[0])
                        y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf


d = Jetson()
