import cv2
import numpy as np
import winsound
import time
import requests

BOT_TOKEN = "8194832927:AAF7Kl9i80DpnnORLga9E9nufNxdXDSXFos"
CHAT_ID = "5250208201"

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message
    }
    requests.post(url, data=data)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")  

names = ["Naveen"]

cap = cv2.VideoCapture(0)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

last_alert_time = 0

while True:
    diff = cv2.absdiff(frame1, frame2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 2000:
            continue
        motion_detected = True
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(frame1, "Motion Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)
        

        if confidence < 70:
            name = names[label]
            color = (0, 255, 0)                              
        else:
            name = "Unknown"
            color = (0, 0, 255)

            
            if time.time() - last_alert_time > 5:
                winsound.Beep(1000, 700)
                print("⚠ ALERT: Unknown face detected!")
                send_telegram_alert("⚠ ALERT: Unknown face detected!")

                last_alert_time = time.time()

        cv2.rectangle(frame1, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame1, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Live Security System", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 