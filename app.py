import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Harmony Analyzer", layout="centered")
st.title("💠 Facial Harmony & PSL Analyzer")

uploaded_file = st.file_uploader("Upload a clear front-facing image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    h, w, _ = img_array.shape

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
    results = face_mesh.process(img_array)

    if not results.multi_face_landmarks:
        st.error("No face detected. Try another image.")
    else:
        landmarks = results.multi_face_landmarks[0].landmark

        def get_point(i):
            return np.array([landmarks[i].x * w, landmarks[i].y * h])

        # Key landmarks
        pts = {
            "left_cheek": get_point(234),
            "right_cheek": get_point(454),
            "chin": get_point(152),
            "forehead": get_point(10),
            "glabella": get_point(168),
            "nose_tip": get_point(1),
            "nose_base_left": get_point(49),
            "nose_base_right": get_point(279),
            "mouth_left": get_point(61),
            "mouth_right": get_point(291),
            "upper_lip": get_point(13),
            "philtrum": get_point(0),
            "pupil_left": get_point(468),
            "pupil_right": get_point(473),
            "eye_outer_left": get_point(33),
            "eye_outer_right": get_point(263),
            "eye_inner_left": get_point(133),
            "eye_inner_right": get_point(362),
        }

        # Measurements
        face_width = np.linalg.norm(pts["left_cheek"] - pts["right_cheek"])
        face_height = np.linalg.norm(pts["forehead"] - pts["chin"])
        jaw_width = face_width
        nose_width = np.linalg.norm(pts["nose_base_left"] - pts["nose_base_right"])
        mouth_width = np.linalg.norm(pts["mouth_left"] - pts["mouth_right"])
        philtrum_height = np.linalg.norm(pts["philtrum"] - pts["upper_lip"])
        chin_height = np.linalg.norm(pts["upper_lip"] - pts["chin"])

        upper_third = np.linalg.norm(pts["forehead"] - pts["glabella"])
        middle_third = np.linalg.norm(pts["glabella"] - pts["nose_tip"])
        lower_third = np.linalg.norm(pts["nose_tip"] - pts["chin"])

        canthal_tilt_deg = np.degrees(np.arctan2(
            pts["eye_outer_right"][1] - pts["eye_inner_right"][1],
            pts["eye_outer_right"][0] - pts["eye_inner_right"][0]
        ))

        # Ideal values
        IDEAL = {
            "fWHR": 1.85,
            "jaw_to_face": 0.85,
            "mouth_to_nose": 1.6,
            "thirds_balance": (1, 1, 1),
            "canthal_tilt": 8,
        }

        def harmony_score(actual, ideal, tolerance=0.15):
            deviation = abs(actual - ideal) / ideal
            return max(0, 1 - deviation / tolerance)

        # Scores
        scores = {
            "fWHR": harmony_score(face_width / face_height, IDEAL["fWHR"]),
            "Jaw/Face Width": harmony_score(jaw_width / face_width, IDEAL["jaw_to_face"]),
            "Mouth/Nose Width": harmony_score(mouth_width / nose_width, IDEAL["mouth_to_nose"]),
            "Facial Thirds Balance": harmony_score(np.std([upper_third, middle_third, lower_third]), 0, tolerance=0.25),
            "Canthal Tilt": harmony_score(canthal_tilt_deg, IDEAL["canthal_tilt"], tolerance=0.5),
        }

        overall_score = round(np.mean(list(scores.values())) * 100, 2)

        if overall_score >= 95:
            psl_tier = "9.5 – 10 (Chad Tier)"
        elif overall_score >= 90:
            psl_tier = "8.5 – 9.5 (Top Model)"
        elif overall_score >= 80:
            psl_tier = "7.5 – 8.5 (Above Average)"
        elif overall_score >= 70:
            psl_tier = "6.5 – 7.5 (Decent Looks)"
        elif overall_score >= 60:
            psl_tier = "5.5 – 6.5 (Average)"
        else:
            psl_tier = "Sub-5.5 (Below Average)"

        st.image(image, caption="Uploaded Face", use_column_width=True)
        st.markdown("### 📊 Facial Harmony Report")
        for k, v in scores.items():
            st.write(f"**{k}**: {round(v * 100, 2)}% match")

        st.markdown(f"### 🧠 Overall Harmony Score: **{overall_score}/100**")
        st.markdown(f"### 🔥 Estimated PSL Tier: **{psl_tier}**")
