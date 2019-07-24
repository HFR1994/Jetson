import cv2
import os
import shutil
import datetime
import time
import platform

def running_on_jetson_nano():
        # To make the same code work on a laptop or on a Jetson Nano, we'll detect when we are running on the Nano
        # so that we can access the camera correctly in that case.
        # On a normal Intel laptop, platform.machine() will be "x86_64" instead of "aarch64"
        return platform.machine() == "aarch64"

def get_jetson_gstreamer_source(capture_width=1280, capture_height=720, display_width=1280, display_height=720,
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

if running_on_jetson_nano():
    # Accessing the camera with OpenCV on a Jetson Nano requires gstreamer with a custom gstreamer source string
    cap = cv2.VideoCapture(get_jetson_gstreamer_source(), cv2.CAP_GSTREAMER)
else:
    # Accessing the camera with OpenCV on a laptop just requires passing in the number of the webcam (usually 0)
    # Note: You can pass in a filename instead if you want to process a video file instead of a live camera stream
    cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

print("'A' si deseas agregar otra cara, 'W' si quieres empezar desde cero")
choice = str(input())

print("Cual es el nombre")
nombre = str(input())


if choice == 'W':
    # Try to remove tree; if failed show an error using try...except on screen
    try:
        shutil.rmtree("img-data")
    except OSError as e:
        print("Advertencia: el directorio no existe, continuando")

if not os.path.exists("img-data"):
    os.mkdir("img-data")

if not os.path.exists("img-data/faces"):
    os.mkdir("img-data/faces")

if os.path.exists("img-data/faces/"+nombre):
    print("Advertencia: Ya esta registrada esa cara, agregando archivos")
else:
    os.mkdir("img-data/faces/"+nombre)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d%H%M%S')
print('img-data/faces/'+nombre+'/'+st+'.mp4')
out = cv2.VideoWriter('img-data/faces/'+nombre+'/'+st+'.mp4', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame', frame)
    else:
        break

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
