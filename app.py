import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw
import face_recognition

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('face_mask_detector.h5')

model = load_model()
classes = ["with_mask", "without_mask", "mask_weared_incorrect"]

st.title("Smart Face Mask Detector")
uploaded_file = st.file_uploader("Upload an image with faces...", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to np array for face_recognition
    img_np = np.array(image)

    # Detect faces
    face_locations = face_recognition.face_locations(img_np)

    if len(face_locations) == 0:
        st.warning("No faces detected.")
    else:
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)

        for top, right, bottom, left in face_locations:
            face_crop = image.crop((left, top, right, bottom)).resize((224,224))
            face_array = np.array(face_crop) / 255.0
            face_batch = np.expand_dims(face_array, axis=0)

            # Predict
            pred_bbox, pred_class = model.predict(face_batch, verbose=0)
            pred_class_name = classes[np.argmax(pred_class[0])]
            pred_confidence = np.max(pred_class[0])

            # Draw results
            color = (0,255,0) if pred_class_name == "with_mask" else (255,0,0)
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            draw.text((left, top-10), f"{pred_class_name} ({pred_confidence:.2f})", fill=color)

        st.image(overlay, caption="Detected faces with mask classification", use_column_width=True)
