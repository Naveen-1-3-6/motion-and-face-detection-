import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0
dataset_path = "dataset"

for person in os.listdir(dataset_path):
    label_map[current_label] = person
    person_path = os.path.join(dataset_path, person)

    for img_name in os.listdir(person_path):
        img = cv2.imread(os.path.join(person_path, img_name), 0)
        faces.append(img)
        labels.append(current_label)

    current_label += 1

recognizer.train(faces, np.array(labels))
recognizer.save("face_model.yml")
