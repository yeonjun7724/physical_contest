import os
import io
import cv2
import base64
import json
import time
import tempfile
import numpy as np
from PIL import Image
import streamlit as st
from openai import OpenAI

# ------------------------------
# OpenAI Client (환경변수 자동 인식)
# ------------------------------
# 기존: client = OpenAI(api_key=OPENAI_API_KEY)
client = OpenAI()   # ← THIS FIXES THE ERROR

# ------------------------------
# 국민체력 정보 (예시값 그대로)
# ------------------------------
KFTA_SCORES = {
    "situp": {
        "male": {
            "20대": [(52, 100), (47, 90), (42, 80), (37, 70), (32, 60), (27, 50), (22, 40), (17, 30), (12, 20), (7, 10), (0, 0)]
        },
        "female": {
            "20대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30), (7, 20), (3, 10), (0, 0)]
        },
    },
    "pushup": {
        "male": {
            "20대": [(42, 100), (37, 90), (32, 80), (27, 70), (22, 60), (17, 50), (12, 40), (8, 30), (4, 20), (2, 10), (0, 0)]
        },
        "female": {
            "20대": [(32, 100), (27, 90), (22, 80), (18, 70), (14, 60), (10, 50), (7, 40), (4, 30), (2, 20), (1, 10), (0, 0)]
        },
    },
}

EXERCISE_NAMES = {
    "situp": "윗몸일으키기",
    "pushup": "팔굽혀펴기",
    "squat": "스쿼트",
    "plank": "플랭크",
    "burpee": "버피",
    "lunge": "런지",
    "jump": "점프",
    "shuttle_run": "왕복 오래달리기",
    "mixed": "혼합 동작",
}

NON_KFTA = {"squat", "lunge", "jump", "burpee", "mixed"}

# ------------------------------
# 영상에서 프레임 추출
# ------------------------------
def extract_frames(video_bytes, num_frames=4, resize=(640, 360)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    duration = frame_count / fps

    idxs = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()
    os.remove(tmp_path)

    return frames, duration


# ------------------------------
# 프레임 → base64
# ------------------------------
def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ------------------------------
# OpenAI VLM 분석
# ------------------------------
def analyze_frames(frames, duration):
    images_payload = [
        {"type": "input_image", "image_url": pil_to_b64(Image.fromarray(f))}
        for f in frames
    ]

    prompt = """
당신은 운동 분석 전문가입니다.
다음 영상의 프레임을 보고 다음 JSON을 반환하세요:

{
 "exercise_key": "...",
 "exercise_name_kr": "...",
 "estimated_reps": 숫자,
 "main_metric": {"type": "reps|seconds", "value": 숫자},
 "posture": "좋음|보통|나쁨",
 "risk": ["항목1", "항목2"]
}
"""

    result = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "user", "content": [{"type": "text", "text": prompt}, *images_payload]}
        ]
    )

    parsed = client.responses.parse(result)
    return parsed.output[0]


# ------------------------------
# KFTA 점수 계산
# ------------------------------
def calc_kfta(exercise_key, gender, age_group, value):
    gender_key = "male" if gender == "남성" else "female"

    if exercise_key in NON_KFTA or exercise_key not in KFTA_SCORES:
        score = min(100, int(value * 2))
        grade = 1 if score >= 90 else 2 if score >= 75 else 3 if score >= 60 else 4 if score >= 45 else 5
        return score, grade, "연구용 평가"

    table = KFTA_SCORES[exercise_key][gender_key][age_group]

    for threshold, sc in table:
        if value >= threshold:
            score = sc
            break

    grade = 1 if score >= 90 else 2 if score >= 75 else 3 if score >= 60 else 4 if score >= 45 else 5
    return score, grade, "국민체력100 기준(예시)"


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="AI 국민체력 분석", layout="wide")

st.title("🏃 AI 기반 국민체력 영상 분석")

with st.sidebar:
    st.header("⚙ 설정")
    age = st.selectbox("연령대", ["20대"])
    gender = st.selectbox("성별", ["남성", "여성"])

st.write("아래에 운동 영상을 업로드하면 자동 분석합니다.")

video_file = st.file_uploader("MP4 영상 업로드", type=["mp4"])

if st.button("분석 시작"):
    if not video_file:
        st.error("먼저 영상을 업로드하세요.")
        st.stop()

    with st.spinner("프레임 추출 중..."):
        frames, duration = extract_frames(video_file.read(), num_frames=4)

    with st.spinner("AI 분석 중..."):
        analysis = analyze_frames(frames, duration)

    st.success("AI 분석 완료!")

    st.write("### 📌 운동 분석 결과")
    st.json(analysis)

    key = analysis["exercise_key"]
    metric = analysis["main_metric"]["value"]

    score, grade, remark = calc_kfta(key, gender, age, metric)

    st.write("### 🏅 국민체력 점수")
    st.metric("점수", f"{score}점")
    st.metric("등급", f"{grade}등급")
    st.caption(remark)
