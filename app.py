import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Facial Harmony Analyzer", layout="centered")
st.title("🧬 Facial Harmony & PSL Analyzer")

uploaded_file = st.file_uploader("Upload a clear front-facing photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    h, w, _ = img_np.shape

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(img_np)

        if not results.multi_face_landmarks:
            st.error("No face detected. Try another image.")
        else:
            lm = results.multi_face_landmarks[0].landmark

            def get(i): return np.array([lm[i].x * w, lm[i].y * h])

            points = {
                "left_cheek": get(234),
                "right_cheek": get(454),
                "chin": get(152),
                "forehead": get(10),
                "glabella": get(168),
                "nose_tip": get(1),
                "nose_left": get(49),
                "nose_right": get(279),
                "mouth_left": get(61),
                "mouth_right": get(291),
                "upper_lip": get(13),
                "philtrum": get(0),
                "eye_outer_r": get(263),
                "eye_inner_r": get(362),
                "eye_outer_l": get(33),
                "eye_inner_l": get(133),
                "eye_top_l": get(159),
                "eye_bot_l": get(145)
            }

            # Ratios
            face_width = np.linalg.norm(points["left_cheek"] - points["right_cheek"])
            face_height = np.linalg.norm(points["forehead"] - points["chin"])
            jaw_width = face_width
            nose_width = np.linalg.norm(points["nose_left"] - points["nose_right"])
            mouth_width = np.linalg.norm(points["mouth_left"] - points["mouth_right"])
            philtrum_height = np.linalg.norm(points["philtrum"] - points["upper_lip"])

            # Thirds
            upper = np.linalg.norm(points["forehead"] - points["glabella"])
            middle = np.linalg.norm(points["glabella"] - points["nose_tip"])
            lower = np.linalg.norm(points["nose_tip"] - points["chin"])

            # Canthal tilt (right eye)
            tilt = np.degrees(np.arctan2(
                points["eye_outer_r"][1] - points["eye_inner_r"][1],
                points["eye_outer_r"][0] - points["eye_inner_r"][0]
            ))

            # Ideal references
            IDEAL = {
                "fWHR": 1.85,
                "jaw_to_face": 0.85,
                "mouth_to_nose": 1.6,
                "thirds": (1, 1, 1),
                "tilt": 8
            }

            def score(val, ideal, tol=0.15):
                return max(0, 1 - abs(val - ideal) / (ideal * tol))

            ratios = {
                "fWHR": face_width / face_height,
                "Jaw/Face": jaw_width / face_width,
                "Mouth/Nose": mouth_width / nose_width,
                "Thirds Balance": np.std([upper, middle, lower]),
                "Canthal Tilt": tilt
            }

            scores = {
                "fWHR": score(ratios["fWHR"], IDEAL["fWHR"]),
                "Jaw/Face": score(ratios["Jaw/Face"], IDEAL["jaw_to_face"]),
                "Mouth/Nose": score(ratios["Mouth/Nose"], IDEAL["mouth_to_nose"]),
                "Thirds Balance": score(ratios["Thirds Balance"], 0, tol=0.25),
                "Canthal Tilt": score(ratios["Canthal Tilt"], IDEAL["tilt"], tol=0.5)
            }

            overall = round(np.mean(list(scores.values())) * 100, 2)
            if overall >= 95:
                tier = "9.5 – 10 (Chad Tier)"
            elif overall >= 90:
                tier = "8.5 – 9.5 (Top Model)"
            elif overall >= 80:
                tier = "7.5 – 8.5 (Above Average)"
            elif overall >= 70:
                tier = "6.5 – 7.5 (Decent Looks)"
            elif overall >= 60:
                tier = "5.5 – 6.5 (Average)"
            else:
                tier = "Sub-5.5 (Below Average)"

            st.image(img, caption="Uploaded Face", use_column_width=True)
            st.markdown("### 🔍 Facial Harmony Scores")
            for k, v in scores.items():
                st.write(f"**{k}**: {round(v*100, 2)}% match")

            st.markdown(f"### 🧠 Overall Score: **{overall}/100**")
            st.markdown(f"### 🔥 Estimated PSL Tier: **{tier}**")
