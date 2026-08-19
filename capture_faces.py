import cv2
import os

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
person_name = "Joan"
save_path = f"dataset/{person_name}"
os.makedirs(save_path, exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        count += 1
        cv2.imwrite(f"{save_path}/{count}.jpg", face)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Collecting Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
