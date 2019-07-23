import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
import re
import face_recognition

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


class Jetson:

    def __init__(self):
        # STEP 1: Train the KNN classifier and save it to disk
        # Once the model is trained and saved, you can skip this step next time.
        print("Training KNN classifier...")
        classifier = self.train("img-data/faces", model_save_path="trained_knn_model.clf", n_neighbors=2)
        print("Training complete!")

        # STEP 2: Using the trained classifier, make predictions for unknown images
        for image_file in os.listdir("img-data/test"):
            full_file_path = os.path.join("img-data/test", image_file)

            print("Looking for faces in {}".format(image_file))

            # Find all people in the image using a trained classifier model
            # Note: You can pass in either a classifier file name or a classifier model instance
            predictions = self.predict(full_file_path, model_path="trained_knn_model.clf")

            # Print results on the console
            for name, (top, right, bottom, left) in predictions:
                print("- Found {} at ({}, {})".format(name, left, top))

    def predict(self, x_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
        if not os.path.isfile(x_img_path) or os.path.splitext(x_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
            raise Exception("Invalid image path: {}".format(x_img_path))

        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        # Load image file and find face locations
        X_img = face_recognition.load_image_file(x_img_path)
        X_face_locations = face_recognition.face_locations(X_img)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    def video_files_in_folder(self, folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(mp4)', f, flags=re.I)]

    def train(self, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        X = []
        y = []

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
