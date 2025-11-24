import streamlit as st
import cv2
import numpy as np
import base64
import requests
import json
from PIL import Image
import io
import time

# -----------------------------------------------------
# OpenAI Vision API 호출 (429/Timeout 방지 버전)
# -----------------------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAME = "gpt-4o-mini"

def encode_frame(frame):
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return base64.b64encode(buf.getvalue()).decode()


def call_openai_vision(messages, retries=6, delay=4):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 1400,
        "temperature": 0.2
    }

    for i in range(retries):
        try:
            res = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=15)
            if res.status_code == 200:
                return res.json()["choices"][0]["message"]["content"]
            elif res.status_code == 429:
                time.sleep(delay)
            else:
                time.sleep(delay)
        except Exception:
            time.sleep(delay)

    raise RuntimeError("⚠ OpenAI API 오류: 여러 번 재시도했지만 응답이 없습니다.")


# -----------------------------------------------------
# 프레임 추출 (3개)
# -----------------------------------------------------
def extract_frames(video_bytes):
    video = np.frombuffer(video_bytes, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(video, cv2.IMREAD_COLOR))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = [int(total * 0.2), int(total * 0.5), int(total * 0.8)]
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames


# -----------------------------------------------------
# VLM 분석 파이프라인
# -----------------------------------------------------
def analyze_exercise(frames):
    images_payload = []

    for f in frames:
        b64 = encode_frame(f)
        images_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    system_prompt = """
당신은 대한민국 국민체력100 공식 기준을 잘 아는 AI 코치입니다.
사용자가 업로드한 영상의 프레임을 보고 어떤 운동인지 분류하고,
동작의 정확도, 반복수 추정, 코칭 포인트, 국민체력100 기준 점수/등급을 출력하세요.

지원 운동 목록:
- 윗몸일으키기(Sit-up)
- 팔굽혀펴기(Push-up)
- 스쿼트(Squat)
- 플랭크(Plank)
- 버피(Burpee)
- 런지(Lunge)
- 제자리 점프 / 스텝박스 점프
- 오래달리기(동작 패턴 보고 가능한 경우 설명)
- 기타 복합 운동: 가장 가까운 운동으로 분류

출력 형식(JSON ONLY):
{
  "exercise_type": "",
  "rep_count_estimated": "",
  "form_quality": "",
  "coach_feedback": "",
  "kfta_score_estimated": "",
  "kfta_grade": ""
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": images_payload}
    ]

    result = call_openai_vision(messages)
    return json.loads(result)


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.set_page_config(
    page_title="AI 국민체력100 운동 분석",
    layout="wide"
)

st.title("💪 AI 기반 국민체력100 운동 분석기 (Demo)")
st.write("영상을 업로드하면 AI가 운동 종류를 인식하고, 자세·반복수·점수·코칭을 제공합니다.")

video = st.file_uploader("🎥 운동 영상 업로드 (mp4)", type=["mp4"])

if video is not None:
    st.video(video)

    if st.button("🔍 운동 분석 실행"):
        video_bytes = video.read()

        with st.spinner("🎬 영상을 분석 중입니다..."):
            frames = extract_frames(video_bytes)

        st.write("### 📸 추출된 영상 프레임")
        cols = st.columns(len(frames))
        for i, f in enumerate(frames):
            cols[i].image(f, caption=f"Frame {i+1}", use_column_width=True)

        with st.spinner("🤖 AI가 운동을 분석 중입니다..."):
            result = analyze_exercise(frames)

        st.success("분석 완료!")

        st.write("## 📝 분석 결과")
        st.json(result)

        st.write("## 🏅 AI 요약")
        st.metric("운동 유형", result["exercise_type"])
        st.metric("예상 반복수", result["rep_count_estimated"])
        st.metric("자세 정확도", result["form_quality"])
        st.metric("예상 점수", f"{result['kfta_score_estimated']} 점")
        st.metric("예상 등급", result["kfta_grade"])

        st.write("## 📘 AI 코치 피드백")
        st.write(result["coach_feedback"])
