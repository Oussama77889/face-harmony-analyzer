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

            pts = {
                "L_cheek": get(234), "R_cheek": get(454), "forehead": get(10),
                "chin": get(152), "glabella": get(168), "nose_tip": get(1),
                "nose_L": get(49), "nose_R": get(279),
                "mouth_L": get(61), "mouth_R": get(291),
                "philtrum": get(0),
                "eye_out_R": get(263), "eye_in_R": get(362)
            }

            fw = np.linalg.norm(pts["L_cheek"] - pts["R_cheek"])
            fh = np.linalg.norm(pts["forehead"] - pts["chin"])
            nw = np.linalg.norm(pts["nose_L"] - pts["nose_R"])
            mw = np.linalg.norm(pts["mouth_L"] - pts["mouth_R"])

            upper = np.linalg.norm(pts["forehead"] - pts["glabella"])
            mid = np.linalg.norm(pts["glabella"] - pts["nose_tip"])
            lower = np.linalg.norm(pts["nose_tip"] - pts["chin"])

            tilt = np.degrees(np.arctan2(
                pts["eye_out_R"][1] - pts["eye_in_R"][1],
                pts["eye_out_R"][0] - pts["eye_in_R"][0]
            ))

            IDEAL = {"fWHR": 1.85, "M/N": 1.6, "tilt": 8}
            def score(v, ideal, tol=0.15):
                return max(0, 1 - abs(v - ideal) / (ideal * tol))

            scores = {
                "fWHR": score(fw / fh, IDEAL["fWHR"]),
                "Mouth/Nose": score(mw / nw, IDEAL["M/N"]),
                "Thirds": score(np.std([upper, mid, lower]), 0, tol=0.25),
                "Canthal tilt": score(tilt, IDEAL["tilt"], tol=0.5),
            }
            overall = round(np.mean(list(scores.values())) * 100, 2)
            tier = (
                "9.5–10 (Chad)" if overall >= 95 else
                "8.5–9.5 (Top Model)" if overall >= 90 else
                "7.5–8.5 (Above Avg)" if overall >= 80 else
                "6.5–7.5 (Decent)" if overall >= 70 else
                "5.5–6.5 (Average)" if overall >= 60 else
                "Sub‑5.5 (Below Avg)"
            )

            st.image(img, caption="Your Upload", use_column_width=True)
            st.markdown("### 📊 Facial Harmony Scores")
            for k, v in scores.items():
                st.write(f"**{k}**: {round(v*100,2)}%")
            st.markdown(f"### 🧠 Overall Score: **{overall}/100**")
            st.markdown(f"### 🔥 PSL Tier: **{tier}**")
