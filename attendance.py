# Online Attendance Face Recognition using opencv


import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

bezos_image = face_recognition.load_image_file("images/bezos1.jpg")
bezos_encoding = face_recognition.face_encodings(bezos_image)[0]

federer_image = face_recognition.load_image_file("images/federer.jpg")
federer_encoding = face_recognition.face_encodings(federer_image)[0]

elon_image = face_recognition.load_image_file("images/elon.jpg")
elon_encoding = face_recognition.face_encodings(elon_image)[0]

srk_image = face_recognition.load_image_file("images/srk.jpg")
srk_encoding = face_recognition.face_encodings(srk_image)[0]

ronaldo_image = face_recognition.load_image_file("images/ronaldo.jpg")
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]

saurav_image = face_recognition.load_image_file("images/saurav.jpg")
saurav_encoding = face_recognition.face_encodings(saurav_image)[0]

messi_image = face_recognition.load_image_file("images/messi.png")
messi_encoding = face_recognition.face_encodings(messi_image)[0]

nadal_image = face_recognition.load_image_file("images/nadal.jpg")
nadal_encoding = face_recognition.face_encodings(nadal_image)[0]

known_face_encoding = [
    nadal_encoding, messi_encoding, srk_encoding, ronaldo_encoding, saurav_encoding, elon_encoding, bezos_encoding, federer_encoding
]

known_faces_names = [
    "Rafael Nadal", "Lionel Messi", "Shah Rukh Khan", "Cristiano Ronaldo", "Saurav Ganguly", "Elon Musk", "Jeff Bezos", "Roger Federer"
]

students = known_faces_names.copy()

face_location = []
face_names = []
face_encodings = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"./AttendanceRecord/{current_date}.csv", 'w+', newline='')
lnwriter = csv.writer(f)
lnwriter.writerow(["Student Name", "Time of Attendance",])

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        name = ""
        matches = face_recognition.compare_faces(
            known_face_encoding, face_encoding)
        face_distance = face_recognition.face_distance(
            known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)
        if name in known_faces_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2

            cv2.putText(frame, name+' Present', bottomLeftCornerOfText,
                        font, fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                print(students)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])
    cv2.imshow("Realtime Face Recogntion Attendence system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
