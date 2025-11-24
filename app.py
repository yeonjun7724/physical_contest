import streamlit as st
import cv2
import numpy as np
import tempfile
import requests
import base64
import json
from PIL import Image
import time

# ============================================================
# OpenAI API 호출 (재시도 포함)
# ============================================================

def call_openai(messages, max_retries=3):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=25)

            if response.status_code == 429:
                time.sleep(2 + attempt)
                continue

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        except Exception:
            if attempt == max_retries - 1:
                raise RuntimeError("❌ OpenAI API가 3회 재시도에도 응답하지 않습니다.")
            time.sleep(1.5)

# ============================================================
# 프레임 추출 (임시파일 방식 — Streamlit Cloud 100% 안정적)
# ============================================================

def extract_frames(video_bytes):
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(video_bytes)
    temp_video.flush()

    cap = cv2.VideoCapture(temp_video.name)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []

    idxs = [
        int(total * 0.15),
        int(total * 0.35),
        int(total * 0.55),
        int(total * 0.75),
        int(total * 0.90),
    ]

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

# ============================================================
# 프레임 → base64 변환
# ============================================================

def pil_to_base64(img):
    _, im_arr = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(im_arr).decode()
    return f"data:image/jpeg;base64,{b64}"

# ============================================================
# AI 분석 호출
# ============================================================

def analyze_frames_with_vlm(frames):
    if len(frames) == 0:
        return {"error": "no_frames"}

    # 이미지 5·10개 제한
    frames = frames[:8]

    images_payload = []
    for img in frames:
        b64 = pil_to_base64(img)
        images_payload.append({"type": "image_url", "image_url": {"url": b64}})

    system_prompt = """
당신은 국민체력100 전문 평가관입니다.
사용자가 업로드한 영상 프레임을 기반으로 다음을 분석하세요:

1) 운동 종류 자동 분류
   - 윗몸일으키기(sit-up)
   - 팔굽혀펴기(push-up)
   - 스쿼트(squat)
   - 플랭크(plank)
   - 런지(lunge)
   - 버피(burpee)
   - 제자리 점프 or 박스 점프
   - 오래 달리기 또는 왕복 달리기
   - 복합 동작(혼합 운동)

2) 운동 동작 평가
   - 신체 정렬
   - 리듬/가동범위
   - 반복동작 여부 파악

3) 국민체력100 기준에 맞는 예상 점수 (0–100)

4) 개선을 위한 코칭 포인트 제공

반드시 JSON 형식으로 출력:
{
  "exercise_type": "...",
  "score": 0~100,
  "analysis": "...",
  "coaching": "..."
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": images_payload}
    ]

    result = call_openai(messages)
    try:
        return json.loads(result)
    except:
        return {"error": "parse_error", "raw": result}

# ============================================================
# Streamlit UI
# ============================================================

def main():
    st.set_page_config(
        page_title="AI 기반 국민체력100 영상 분석",
        layout="centered"
    )

    st.title("🏋️‍♂️ AI 기반 국민체력100 영상 분석 데모")
    st.write("업로드한 **운동 영상(mp4)** 을 VLM이 분석하여 운동 종류를 자동 판별하고, 국민체력100 기준으로 점수화합니다.")

    st.divider()
    st.subheader("📤 영상 업로드")

    video_file = st.file_uploader("mp4 파일을 업로드하세요", type=["mp4"])

    if video_file is None:
        st.info("운동 영상(mp4)을 업로드하면 분석이 시작됩니다.")
        return

    video_bytes = video_file.read()

    st.video(video_bytes)

    # ========================================================
    # 프레임 추출
    # ========================================================
    st.subheader("📸 추출된 프레임")

    frames = extract_frames(video_bytes)

    if len(frames) == 0:
        st.error("❌ 영상을 읽을 수 없습니다. 다른 mp4 파일을 업로드해주세요.")
        st.stop()

    cols = st.columns(len(frames))
    for i, f in enumerate(frames):
        cols[i].image(f, caption=f"Frame {i+1}", use_column_width=True)

    # ========================================================
    # AI 분석
    # ========================================================
    st.subheader("🤖 AI 운동 분석 결과")

    with st.spinner("AI가 운동을 분석하는 중입니다…"):
        result = analyze_frames_with_vlm(frames)

    if "error" in result:
        st.error("❌ 분석 실패. 다시 시도해주세요.")
        st.write(result)
        return

    st.success("분석 완료!")

    st.metric("운동 종류", result["exercise_type"])
    st.metric("예상 점수", f"{result['score']} / 100")

    st.write("### 📊 동작 분석")
    st.write(result["analysis"])

    st.write("### 📝 코칭 포인트")
    st.write(result["coaching"])

    st.divider()
    st.caption("Powered by GPT-4o-mini Vision + Streamlit Cloud")


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":
    main()
