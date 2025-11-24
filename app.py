# ============================================
# 국민체력100 AI VLM 종합 분석 시스템
# 완성판 app.py
# ============================================

import cv2
import base64
import time
import json
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import streamlit as st


# ============================================
# OpenAI 호출 함수 (429 자동 재시도)
# ============================================

def call_openai(messages, model="gpt-4o-mini", max_retries=5):
    api_key = st.secrets["OPENAI_API_KEY"]
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2
    }

    for i in range(max_retries):
        response = requests.post(url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]

        if response.status_code == 429:  # rate limit
            time.sleep(1.2)   # 딜레이 후 재시도
            continue

        # 기타 오류
        st.error(f"API 오류: {response.text}")
        return None

    raise RuntimeError("OpenAI API가 여러 번 재시도했지만 응답하지 않습니다.")


# ============================================
# 프레임 추출 함수 (8~12 프레임)
# ============================================

def extract_frames(video_bytes, num_frames=10):
    """mp4 바이트 → OpenCV 영상 → 프레임 추출"""

    # 바이너리를 임시 파일로 저장
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (512, 288))
        frames.append(frame)

    cap.release()
    return frames


# ============================================
# 프레임 → base64 이미지 변환
# ============================================

def pil_to_b64(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ============================================
# 프레임 기반 VLM 분석
# ============================================

def analyze_frames(frames):
    """운동 분류 + 반복횟수 추정 + 자세평가"""

    images_payload = []

    # 이미지 10개를 multi-modal 메시지로 구성
    for f in frames:
        b64 = pil_to_b64(Image.fromarray(f))
        images_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    system_prompt = """
당신은 국민체력100 전문가이자 Vision-Language 모델입니다.
10장의 프레임을 보고 다음 항목을 JSON 으로만 출력하세요.

{
 "exercise_type": "situp | pushup | squat | plank | burpee | lunge | jump | shuttle_run | unknown",
 "estimated_reps": 숫자,
 "posture_score": 0~40,
 "tempo": "slow | steady | fast",
 "stability": "low | medium | high",
 "risk_flags": ["무릎 흔들림", "허리 굽힘", ...]
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": images_payload}
    ]

    result = call_openai(messages)
    return json.loads(result)


# ============================================
# 국민체력100 점수 계산
# ============================================

def score_kfta(exercise_type, reps, posture_score):
    """운동별 기준 점수 계산"""

    # ------------------------------
    # 국민체력100 간이 점수표 (임시)
    # ------------------------------
    table = {
        "situp": 30,
        "pushup": 40,
        "squat": 40,
        "burpee": 30,
        "lunge": 30,
        "jump": 50,
        "shuttle_run": 40,
    }

    if exercise_type not in table:
        return 0, 5

    max_reps = table[exercise_type]

    performance_score = min(reps / max_reps * 60, 60)
    total = int(min(performance_score + posture_score, 100))

    if total >= 90: grade = 1
    elif total >= 75: grade = 2
    elif total >= 60: grade = 3
    elif total >= 45: grade = 4
    else: grade = 5

    return total, grade


# ============================================
# Streamlit UI
# ============================================

def main():
    st.set_page_config(page_title="국민체력100 AI 분석", layout="wide")
    st.title("🏋️‍♂️ 국민체력100 AI 운동 분석기 (GPT-4o-mini Vision)")

    st.markdown("mp4 영상을 업로드하면 AI가 **운동 종류, 반복 횟수, 자세 평가, 국민체력100 점수**를 분석합니다.")

    video = st.file_uploader("운동 영상 업로드 (mp4)", type=["mp4"])

    if video:
        video_bytes = video.read()

        st.subheader("📸 1) 영상에서 대표 프레임 추출")
        frames = extract_frames(video_bytes)

        if len(frames) == 0:
            st.error("프레임 추출 실패. 다른 영상으로 시도해주세요.")
            st.stop()

        cols = st.columns(min(len(frames), 5))
        for i, f in enumerate(frames[:5]):
            cols[i].image(f, caption=f"Frame {i+1}", use_container_width=True)

        st.subheader("🤖 2) AI VLM 분석 중…")
        with st.spinner("GPT-4o-mini가 영상 분석 중…"):
            result = analyze_frames(frames)

        st.json(result)

        # 국민체력100 점수 계산
        exercise_type = result["exercise_type"]
        reps = result["estimated_reps"]
        posture_score = result["posture_score"]

        kfta_score, grade = score_kfta(exercise_type, reps, posture_score)

        st.subheader("🏅 3) 국민체력100 자동 점수 산출")
        st.metric("총점", f"{kfta_score}/100")
        st.metric("예상등급", f"{grade} 등급")

        st.success("분석 완료!")


if __name__ == "__main__":
    main()
