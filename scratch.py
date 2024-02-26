import face_recognition
import numpy as np
import csv
import cv2
from datetime import datetime
import sys


video_capture = cv2.VideoCapture(0)
#loading the known faces
saumi_image = face_recognition.load_image_file("faces/saumi.jpeg")
#encoding the faces
saumi_encoding = face_recognition.face_encodings(saumi_image)[0]
harshit_image = face_recognition.load_image_file("faces/harshit.jpg")
harshit_encoding = face_recognition.face_encodings(harshit_image)[0]

known_face_encoding = [saumi_encoding, harshit_encoding]
known_face_name = ["saumi", "harshit"]


# list of expected students
students = known_face_name.copy()

face_encodings = []
face_locations = []

# current date and time to show on the screen

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

# creating a csv file

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # recognising the faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_name[best_match_index]

        # adding the name of the person
        if name in known_face_name:
            font = cv2.FONT_HERSHEY_DUPLEX
            bottomLeft = (10,100)
            fontScale = 1.5
            fontColor = (200, 0, 0)
            fontThick = 1
            lineType = 2
            cv2.putText(frame, name + " present", bottomLeft, font, fontScale, fontColor, fontThick, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M-%S")
                lnwriter.writerow([name, current_time])





    cv2.imshow("attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()