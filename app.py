import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

st.title("Smart Face Mask Detector - Only on Faces")
model = tf.keras.models.load_model('face_mask_detector.h5')
classes = ["with_mask","without_mask","mask_weared_incorrect"]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

run = st.checkbox('Start Webcam')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x,y,w,h) in faces:
        face_crop = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (224,224)) / 255.0
        pred_bbox, pred_class = model.predict(np.expand_dims(face_resized,0), verbose=0)
        pred_class = pred_class[0]
        class_name = classes[np.argmax(pred_class)]
        color = (0,255,0) if class_name=="with_mask" else (0,0,255)

        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        frame = cv2.putText(frame, class_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()
