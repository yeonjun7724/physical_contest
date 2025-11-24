import streamlit as st
import cv2
import base64
import json
import numpy as np
from PIL import Image
from io import BytesIO
import requests

# ------------------------------------------------------------
# GPT-4o-mini Vision 호출 함수 (429 방지)
# ------------------------------------------------------------
def call_openai(messages):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.secrets['OPENAI_API_KEY']}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 900
    }

    # --- 재시도 로직 (429 방지) ---
    for attempt in range(3):
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 429:
            st.warning("API 사용량이 몰려 재시도 중입니다… (429 Too Many Requests)")
            import time
            time.sleep(3)
            continue

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise RuntimeError("OpenAI API가 3회 재시도에도 응답하지 않습니다.")

# ------------------------------------------------------------
# 이미지(base64로 변환)
# ------------------------------------------------------------
def pil_to_base64(pil_img):
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

# ------------------------------------------------------------
# 영상 → 프레임 n개 추출
# ------------------------------------------------------------
def extract_frames(video_bytes, n_frames=4):
    np_video = np.frombuffer(video_bytes, np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(np_video, cv2.IMREAD_COLOR))

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_list = np.linspace(0, frame_count - 1, n_frames).astype(int)

    for idx in idx_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

# ------------------------------------------------------------
# 프레임 기반 운동 분석
# ------------------------------------------------------------
def analyze_frames_with_vlm(frames):
    images_payload = []

    for img in frames:
        pil_img = Image.fromarray(img)
        b64 = pil_to_base64(pil_img)
        images_payload.append({
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{b64}"
        })

    system_prompt = """
당신은 AI 기반 체력측정 전문가입니다.
입력된 여러 이미지(프레임)를 보고 어떤 운동인지 판단하고,
국민체력100 기준으로 평가하세요.

지원해야 하는 운동 종류:
- 윗몸일으키기 (Sit-up)
- 팔굽혀펴기 (Push-up)
- 스쿼트 (Squat)
- 플랭크 (Plank)
- 버피 (Burpee)
- 런지 (Lunge)
- 제자리 점프 / 스텝박스 점프
- 오래달리기(왕복달리기)
- 종합 체력테스트 동작

반드시 JSON 형식으로 답변:
{
 "detected_exercise": "운동명",
 "explanation": "판단 근거",
 "score": {
     "total_score": 숫자,
     "grade": "등급",
     "detail": "세부 내용"
 }
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": images_payload}
    ]

    result = call_openai(messages)
    return json.loads(result)

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
def main():
    st.set_page_config(page_title="AI 체력측정 (VLM)", layout="wide")
    st.title("💪 AI 기반 국민체력 100 자동 측정기")
    st.write("업로드한 영상에서 자동으로 운동 종류를 인식하고 점수/등급을 분석합니다.")

    uploaded = st.file_uploader("📤 운동 영상 업로드 (mp4)", type=["mp4"])

    if not uploaded:
        st.info("운동 영상을 업로드하세요.")
        return

    # 영상 처리
    video_bytes = uploaded.read()

    st.subheader("1) 영상 프레임 미리보기")
    frames = extract_frames(video_bytes, n_frames=4)

    cols = st.columns(4)
    for i, f in enumerate(frames):
        cols[i].image(f, caption=f"Frame {i+1}", use_container_width=True)

    st.subheader("2) AI 분석 결과")
    with st.spinner("🔥 VLM이 운동을 분석하는 중…"):
        result = analyze_frames_with_vlm(frames)

    # 결과 표시
    st.success("분석 완료!")

    st.json(result)

    st.subheader("3) 요약 결과")
    st.metric("감지된 운동", result["detected_exercise"])
    st.metric("총점", f"{result['score']['total_score']}점")
    st.metric("예상 등급", result["score"]["grade"])

    st.write("### 세부 분석 리포트")
    st.write(result["score"]["detail"])

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
