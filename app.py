import os
import cv2
import base64
import time
import json
import requests
import numpy as np
import streamlit as st
from PIL import Image


# ============================================================
# 0. OpenAI API 호출(429 방지용 재시도 포함)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def call_openai(messages, max_retries=5):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 1800,
        "temperature": 0.2
    }

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 429:
            wait = 2 * (attempt + 1)
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise RuntimeError("OpenAI 429 rate limit — 재시도 실패")


# ============================================================
# 1. 프레임 추출 함수 (4프레임)
# ============================================================

def extract_frames(video_bytes, num_frames=4, size=(384, 384)):
    np_bytes = np.frombuffer(video_bytes, np.uint8)
    video = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    temp_path = "temp_input.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    idxs = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
            frames.append(frame)

    cap.release()
    return frames


def pil_to_base64(img):
    buf = st.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ============================================================
# 2. VLM 분류 + 자세 분석
# ============================================================

def analyze_frames_with_vlm(frames):
    images_payload = []

    for img in frames:
        b64 = pil_to_base64(Image.fromarray(img))
        images_payload.append({"type": "image_url", "image_url": {"url": b64}})

    system_prompt = """
당신은 한국 국민체력100 전문가이며, 영상 사진을 기반으로 운동 종류와 자세를 분석합니다.

출력은 JSON 하나만!
{
 "exercise_type": "...",   // sit-up, push-up, squat, plank, burpee, lunge, shuttle_run, jump 등
 "key_points": "...",       // 핵심 자세 설명
 "risk": "...",             // 부상 가능성
 "score_raw": 0-100         // 대략적 수행 수준(추정)
}
"""

    user_prompt = "아래 프레임을 기반으로 운동 종류와 수행 상태를 분석하세요."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}] + images_payload}
    ]

    result = call_openai(messages)
    return json.loads(result)


# ============================================================
# 3. Streamlit UI
# ============================================================

def main():
    st.set_page_config(page_title="AI 국민체력100 자동 분석기", layout="centered")

    st.title("🏋️ AI 체력측정 자동 분석 (VLM 기반)")
    st.write("한국 국민체력100 기준으로 영상 속 운동을 자동 인식하고 분석합니다.")

    video = st.file_uploader("운동 영상(mp4) 업로드", type=["mp4"])

    if video is None:
        st.info("운동 영상을 업로드해주세요.")
        return

    if st.button("🚀 분석 시작하기", type="primary"):
        video_bytes = video.read()

        st.subheader("1) 대표 프레임 추출")
        frames = extract_frames(video_bytes)

        col = st.columns(len(frames))
        for i, f in enumerate(frames):
            col[i].image(f, caption=f"Frame {i+1}")

        with st.spinner("AI가 운동을 분석하고 있습니다…"):
            result = analyze_frames_with_vlm(frames)

        st.success("분석 완료!")

        st.subheader("2) 분석 결과 (JSON)")
        st.json(result)

        st.subheader("3) 자연어 요약 리포트")
        st.write(f"""
### 🔍 운동 분류  
- **운동 종류:** {result['exercise_type']}

### 👍 주요 포인트  
{result['key_points']}

### ⚠️ 부상 위험  
{result['risk']}

### ⭐ 수행 점수 (추정)  
**{result['score_raw']} / 100**
        """)


# ============================================================
if __name__ == "__main__":
    if OPENAI_API_KEY is None:
        st.error("❗ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    main()
