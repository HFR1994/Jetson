import face_recognition
import cv2
import numpy as np
import platform
import sqlite3

conn = None


# noinspection SqlResolve
class Jetson:

    def __init__(self):
        self.conn = None
        self.cursor = None
        self.weights1 = np.array([[0.8660962, 0.71246127, 0.76005742, 0.30884078],
                                  [0.39601987, 0.92665683, 0.39860433, 0.35735376],
                                  [0.11932107, 0.81509026, 0.16314202, 0.68627833],
                                  [0.82622397, 0.5208003, 0.06846836, 0.97751566],
                                  [0.351693, 0.77674822, 0.08387581, 0.80589276],
                                  [0.16318295, 0.01247562, 0.59042992, 0.99020929],
                                  [0.4205685, 0.65878184, 0.13458912, 0.10118183],
                                  [0.50437472, 0.76931219, 0.11455795, 0.74569572],
                                  [0.85959587, 0.65829408, 0.51407573, 0.27384887],
                                  [0.70136795, 0.80423257, 0.43906047, 0.13545913],
                                  [0.5907796, 0.40477479, 0.25126451, 0.83436658],
                                  [0.76622969, 0.39216146, 0.49416206, 0.65635806],
                                  [0.47482714, 0.12257853, 0.7545487, 0.61990916],
                                  [0.63397908, 0.73267673, 0.27486918, 0.9174663],
                                  [0.70352831, 0.96859642, 0.31033597, 0.54913333],
                                  [0.71002906, 0.77127421, 0.98505932, 0.50932212],
                                  [0.84748761, 0.31293774, 0.74076195, 0.11205275],
                                  [0.50969904, 0.31653005, 0.46563852, 0.23473548],
                                  [0.13338296, 0.9475233, 0.39606029, 0.5741068],
                                  [0.38737283, 0.81592018, 0.93385854, 0.65267405],
                                  [0.96432777, 0.46900134, 0.73545839, 0.70371038],
                                  [0.88387017, 0.2773811, 0.54856548, 0.06654055],
                                  [0.00620755, 0.94794161, 0.64926182, 0.41585254],
                                  [0.81606323, 0.59423344, 0.17193834, 0.57584688],
                                  [0.48813775, 0.3396094, 0.45693831, 0.58755712],
                                  [0.12821064, 0.26740121, 0.01154033, 0.41791864],
                                  [0.08358671, 0.89449442, 0.4052907, 0.1385597],
                                  [0.37896797, 0.85967368, 0.14519192, 0.60017049],
                                  [0.33026456, 0.51963824, 0.0949977, 0.16346447],
                                  [0.20023444, 0.74989383, 0.05459469, 0.78509702],
                                  [0.88033321, 0.72888416, 0.69610014, 0.22849774],
                                  [0.85507515, 0.63172684, 0.28607973, 0.16983876],
                                  [0.15526344, 0.41270761, 0.6325964, 0.84202891],
                                  [0.19898096, 0.14586511, 0.65058443, 0.58381579],
                                  [0.83354533, 0.23683288, 0.79010501, 0.8990508],
                                  [0.33834145, 0.36473754, 0.65648128, 0.08327609],
                                  [0.75192957, 0.25713725, 0.54402269, 0.72453575],
                                  [0.79881519, 0.54257427, 0.24210819, 0.75891175],
                                  [0.50888656, 0.56656898, 0.94218952, 0.17964441],
                                  [0.99721313, 0.19370849, 0.5450988, 0.36330728],
                                  [0.0423237, 0.4267712, 0.65672522, 0.30527733],
                                  [0.41815339, 0.61172986, 0.09962784, 0.92472474],
                                  [0.13159761, 0.94875209, 0.79333644, 0.09614505],
                                  [0.24856003, 0.45488412, 0.63871632, 0.51522879],
                                  [0.69817273, 0.98660662, 0.31047233, 0.7727561],
                                  [0.11587899, 0.88256493, 0.68208621, 0.40495078],
                                  [0.98283445, 0.14168748, 0.73105495, 0.80930612],
                                  [0.70721484, 0.07028835, 0.20444406, 0.48537216],
                                  [0.32027893, 0.60445239, 0.44318611, 0.46184042],
                                  [0.41651251, 0.94410017, 0.15280285, 0.51525342],
                                  [0.7831936, 0.78070666, 0.12443, 0.76119398],
                                  [0.49059684, 0.48527326, 0.68080806, 0.09914734],
                                  [0.00837381, 0.62698531, 0.86193217, 0.06324836],
                                  [0.42430072, 0.53658546, 0.84140961, 0.16209644],
                                  [0.58361193, 0.59284642, 0.84638284, 0.40617292],
                                  [0.29106714, 0.69766501, 0.01904218, 0.13298855],
                                  [0.79557066, 0.19601236, 0.98154739, 0.48288952],
                                  [0.61892494, 0.34204855, 0.46368993, 0.0761327],
                                  [0.12570125, 0.27008698, 0.81389709, 0.17710306],
                                  [0.53919321, 0.08765035, 0.42821626, 0.24759364],
                                  [0.55498027, 0.6226843, 0.43444922, 0.57460322],
                                  [0.15583762, 0.40050565, 0.73214745, 0.56164234],
                                  [0.64601244, 0.43791889, 0.28884391, 0.41672856],
                                  [0.55609928, 0.0657116, 0.88161252, 0.77841962],
                                  [0.91260501, 0.07518364, 0.48608721, 0.07536418],
                                  [0.54839744, 0.96750926, 0.34109736, 0.69888752],
                                  [0.98715655, 0.40241846, 0.26126871, 0.32814979],
                                  [0.59638996, 0.89022459, 0.42994844, 0.57480067],
                                  [0.9809499, 0.85756184, 0.74854451, 0.96958842],
                                  [0.33126917, 0.56349114, 0.89586938, 0.99816284],
                                  [0.97211371, 0.63020682, 0.64319616, 0.82518683],
                                  [0.84793194, 0.81919661, 0.0956502, 0.40409894],
                                  [0.53158704, 0.55889357, 0.21997608, 0.59075804],
                                  [0.13646132, 0.38624945, 0.55629379, 0.50442223],
                                  [0.61793867, 0.8929028, 0.85628748, 0.17770272],
                                  [0.07640806, 0.66857765, 0.91286576, 0.38901002],
                                  [0.12540763, 0.66695399, 0.75806799, 0.8777354],
                                  [0.22648953, 0.15498522, 0.73123273, 0.03725083],
                                  [0.94377178, 0.22701481, 0.93982158, 0.33166129],
                                  [0.37711727, 0.12654979, 0.27866809, 0.24972245],
                                  [0.66810065, 0.59032153, 0.6293717, 0.71654419],
                                  [0.7753518, 0.17476703, 0.55272779, 0.01456204],
                                  [0.17396387, 0.2047349, 0.467756, 0.98596254],
                                  [0.07690796, 0.56845502, 0.71201588, 0.86846857],
                                  [0.93075293, 0.53498126, 0.3688333, 0.49975906],
                                  [0.9310697, 0.67678875, 0.69542682, 0.84838363],
                                  [0.52697294, 0.20907638, 0.7125502, 0.55086169],
                                  [0.82885715, 0.80822805, 0.27327992, 0.10565825],
                                  [0.27622621, 0.99253891, 0.80316614, 0.41891096],
                                  [0.24020763, 0.38104874, 0.39514982, 0.72352643],
                                  [0.98895839, 0.07804445, 0.23150162, 0.46613511],
                                  [0.1230104, 0.83088317, 0.46121332, 0.20757046],
                                  [0.9781336, 0.08469821, 0.37723667, 0.33682532],
                                  [0.58944375, 0.74630678, 0.39732204, 0.58695133],
                                  [0.06793803, 0.21558462, 0.09373861, 0.98667255],
                                  [0.0365842, 0.86018132, 0.08706251, 0.64539019],
                                  [0.40811659, 0.84431233, 0.90724585, 0.23925609],
                                  [0.02752822, 0.63391106, 0.95135742, 0.33663185],
                                  [0.91726323, 0.49093665, 0.41551579, 0.49936464],
                                  [0.93488024, 0.37143381, 0.38791237, 0.0182133],
                                  [0.65454456, 0.94343867, 0.57812628, 0.12294923],
                                  [0.01716035, 0.61581502, 0.71077752, 0.63277285],
                                  [0.41439258, 0.11294107, 0.25251583, 0.68223954],
                                  [0.44795736, 0.35962532, 0.24440749, 0.81953057],
                                  [0.30198561, 0.76189125, 0.71004384, 0.87848621],
                                  [0.92717213, 0.20458221, 0.29731972, 0.69907745],
                                  [0.42548534, 0.62356126, 0.79645418, 0.7792036],
                                  [0.96739556, 0.83218276, 0.05225696, 0.45063353],
                                  [0.12777257, 0.7398407, 0.03062148, 0.715297],
                                  [0.89325362, 0.89075798, 0.65331472, 0.87879021],
                                  [0.27617417, 0.29373583, 0.14841641, 0.74278384],
                                  [0.45770646, 0.71477877, 0.17963732, 0.70688384],
                                  [0.1536551, 0.32545629, 0.79122625, 0.44541001],
                                  [0.61457593, 0.37637111, 0.83010742, 0.17266169],
                                  [0.33140628, 0.1564524, 0.7521917, 0.53732065],
                                  [0.87798353, 0.36325538, 0.78276252, 0.57313567],
                                  [0.0978097, 0.35163371, 0.8424237, 0.35650276],
                                  [0.17589265, 0.38210746, 0.96040424, 0.34448024],
                                  [0.8707813, 0.73736275, 0.57273151, 0.48786945],
                                  [0.935857, 0.54213215, 0.00869735, 0.48468809],
                                  [0.60539094, 0.35724766, 0.02425848, 0.80875857],
                                  [0.01269647, 0.14291532, 0.69821624, 0.66495227],
                                  [0.1365963, 0.4972635, 0.81291405, 0.41033329],
                                  [0.09905066, 0.25872995, 0.910639, 0.46330269],
                                  [0.11117042, 0.4690765, 0.22033914, 0.89595441],
                                  [0.6337297, 0.46820469, 0.57334525, 0.30343528],
                                  [0.07260391, 0.5216425, 0.29330155, 0.94708116],
                                  [0.76015872, 0.36012621, 0.25510717, 0.91771353]])
        self.weights2 = np.array([[0.24010892],
                                  [0.83526793],
                                  [0.40150045],
                                  [0.00361562]])

    def hash_value(self, foto):

        if len(foto) == 0:
            return 0
        else:
            layer1 = self.activation(np.dot(foto, self.weights1))
            output = self.activation(np.dot(layer1, self.weights2))
            return np.dot(output, np.linalg.norm([foto], ord=3, axis=1))

    def get_data(self):
        # TODO Generate connection to server
        return [
            {"hash": 8792184, "name": "Hector Flores"},
            {"hash": 65456, "name": "Alejandro Cruz"},
            {"hash": 435647, "name": "Argenis Cervantes"}
        ]

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
        # Get access to the webcam. The method is different depending on if this is running on a laptop or a Jetson Nano.
        if self.running_on_jetson_nano():
            # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
            video_capture = cv2.VideoCapture(self.get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
        else:
            # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
            # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
            video_capture = cv2.VideoCapture(0)

        process_this_frame = True

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

                # top, right, bottom, left = face_location
                #
                # top *= 4
                # right *= 4
                # bottom *= 4
                # left *= 4
                #
                # cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                i = i+1

                val = self.hash_value(face_encoding)
                if val != 0:
                    print("Persona {} su hash es {}".format(i, val))
                    # Draw a label with a name below the face
                    # cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    # font = cv2.FONT_HERSHEY_DUPLEX
                    # cv2.putText(frame, str(val), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Display the final frame of video with boxes drawn around each detected fames
            # cv2.imshow('Video', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


d = Jetson()
d.principal()
