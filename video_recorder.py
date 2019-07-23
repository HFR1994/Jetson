import cv2
import os
import shutil
import datetime
import time

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
