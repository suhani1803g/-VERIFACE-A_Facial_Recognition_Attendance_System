import cv2
import os
import sys
import numpy as np
import face_recognition
import cvzone
import csv
from datetime import datetime

# -------------------------------
# SAFE BASE DIRECTORY
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# STUDENT DATABASE 
# -------------------------------
students = {
    "2312": "Suhani Gupta"
}

student_id = "2312"   # Change if needed
student_name = students[student_id]

# -------------------------------
# CREATE ATTENDANCE FILE IF NOT EXISTS
# -------------------------------
attendance_path = os.path.join(BASE_DIR, "attendance.csv")

if not os.path.exists(attendance_path):
    with open(attendance_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Name", "Date", "Time"])

# -------------------------------
# FUNCTION TO MARK ATTENDANCE
# -------------------------------
def mark_attendance(s_id, s_name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(attendance_path, "r") as f:
        data = f.readlines()
        for line in data:
            if s_id in line and date in line:
                return "Already Marked Today"

    with open(attendance_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([s_id, s_name, date, time])

    return "Attendance Marked Successfully"

# -------------------------------
# LOAD BACKGROUND
# -------------------------------
background_path = os.path.join(BASE_DIR, "resources", "background.png")

if not os.path.exists(background_path):
    print("❌ background.png not found at:", background_path)
    sys.exit(1)

imageBackground = cv2.imread(background_path)

# -------------------------------
# LOAD STUDENT IMAGE
# -------------------------------
student_path = os.path.join(BASE_DIR, "resources", "student.png")

if not os.path.exists(student_path):
    print("❌ student.png not found at:", student_path)
    sys.exit(1)

imgStudent = cv2.imread(student_path)
imgStudent = cv2.resize(imgStudent, (216, 216))

# -------------------------------
# ENCODE STUDENT FACE
# -------------------------------
student_image_rgb = cv2.cvtColor(imgStudent, cv2.COLOR_BGR2RGB)
student_encoding = face_recognition.face_encodings(student_image_rgb)[0]

# -------------------------------
# SETUP WEBCAM
# -------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

attendance_message = ""

print("✅ System Started")

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    success, img = cap.read()
    if not success:
        break

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

    faceLocations = face_recognition.face_locations(imgSmall)
    faceEncodings = face_recognition.face_encodings(imgSmall, faceLocations)

    imgOutput = imageBackground.copy()
    imgOutput[162:162+480, 55:55+640] = img

    if faceLocations:
        for encodeFace, faceLoc in zip(faceEncodings, faceLocations):

            matches = face_recognition.compare_faces([student_encoding], encodeFace)
            faceDistance = face_recognition.face_distance([student_encoding], encodeFace)

            if matches[0] and faceDistance[0] < 0.5:

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

                bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                imgOutput = cvzone.cornerRect(imgOutput, bbox, rt=0)

                imgOutput[175:175+216, 909:909+216] = imgStudent

                attendance_message = mark_attendance(student_id, student_name)

                cv2.putText(imgOutput, f"Name: {student_name}",
                            (850, 100),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)

                cv2.putText(imgOutput, f"ID: {student_id}",
                            (850, 130),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)

                cv2.putText(imgOutput, attendance_message,
                            (750, 160),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.7,
                            (255, 0, 0),
                            2)

            else:
                cv2.putText(imgOutput, "Unknown Face",
                            (850, 150),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1,
                            (0, 0, 255),
                            2)

    else:
        cv2.putText(imgOutput, "No Face",
                    (850, 150),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 255),
                    2)

    cv2.imshow("VeriFace Attendance System", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()