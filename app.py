import streamlit as st
import cv2
import numpy as np
import base64
import requests
import json
from io import BytesIO
from PIL import Image

# ============================================================
# 1) OpenAI REST API 호출 함수 (gpt-4o-mini)
# ============================================================

def call_openai(messages, model="gpt-4o-mini"):
    api_key = st.secrets["OPENAI_API_KEY"]

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.2,
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ============================================================
# 2) 영상에서 N개의 프레임 추출
# ============================================================

def extract_frames(video_bytes, num_frames=8):
    file_bytes = np.frombuffer(video_bytes, np.uint8)
    video = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if video is None:
        # mp4는 imdecode가 아니고 VideoCapture 필요
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(temp_path)
    else:
        cap = cv2.VideoCapture()

    cap.open("temp_video.mp4")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx_list = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []

    for idx in idx_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


# ============================================================
# 3) VLM 분석 (프레임 + 운동 분류 + 코칭)
# ============================================================

def analyze_frames_with_vlm(frames):
    # 프레임을 base64 이미지로 변환하여 LLM에 전달
    images_payload = []
    for f in frames:
        img = Image.fromarray(f)
        buf = BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        images_payload.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    system_prompt = """
당신은 운동 분석 및 체력측정 전문가입니다.
아래 프레임들을 보고 어떤 운동인지 자동으로 분류하고,
자세 오류, 템포, 관절 가동 범위, 반동 여부, 신체정렬 등을 평가하세요.

지원 운동 리스트:
1) Sit-up 2) Push-up 3) Squat 4) Plank 5) Burpee 6) Lunge
7) Shuttle-run(왕복달리기) 8) Jump/Step-box Jump 9) 복합 체력측정 동작

출력 형식(JSON):
{
  "exercise_type": "운동명",
  "analysis": "자세 평가 요약",
  "recommendation": "개선 포인트",
  "score_components": {
      "posture": 0~40,
      "tempo": 0~20,
      "range_of_motion": 0~20,
      "stability": 0~20
  }
}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": images_payload}
    ]

    result = call_openai(messages)
    return json.loads(result)


# ============================================================
# 4) 점수 계산 (국민체력100 스타일)
# ============================================================

def score_kfta(result_json):
    comp = result_json["score_components"]

    total = comp["posture"] + comp["tempo"] + comp["range_of_motion"] + comp["stability"]

    if total >= 85:
        grade = "상"
    elif total >= 70:
        grade = "중"
    else:
        grade = "하"

    return total, grade


# ============================================================
# 5) Streamlit UI
# ============================================================

def main():
    st.set_page_config(page_title="AI 체력측정 분석기", layout="wide")

    st.title("🏋️‍♂️ AI 영상 기반 체력측정 분석기 (Sit-up, Push-up, Squat, Plank, Burpee 등)")

    st.write("업로드한 영상을 기반으로 **운동을 자동으로 분류하고**, 자세 분석 + 점수화(국민체력100 스타일)를 수행합니다.")

    st.subheader("1. 영상 업로드")
    video = st.file_uploader("MP4 파일 업로드", type=["mp4"])

    if video is not None:
        video_bytes = video.read()

        st.video(video_bytes)

        st.subheader("2. 분석 실행")
        if st.button("🚀 분석 시작"):
            with st.spinner("영상에서 프레임 추출 중…"):
                frames = extract_frames(video_bytes, num_frames=8)

            st.success(f"프레임 {len(frames)}개 추출 완료")

            st.subheader("샘플 프레임 확인")
            cols = st.columns(4)
            for i, f in enumerate(frames[:4]):
                with cols[i]:
                    st.image(f, caption=f"Frame {i+1}")

            with st.spinner("AI VLM이 운동을 분석하는 중…"):
                result = analyze_frames_with_vlm(frames)

            st.success("분석 완료!")

            # 결과 출력
            st.subheader("3. AI 분석 결과")
            st.json(result)

            total, grade = score_kfta(result)

            st.subheader("4. 점수 결과 (국민체력100 스타일)")
            st.metric("총점", f"{total} / 100")
            st.metric("등급", grade)

            st.subheader("5. AI 코치 피드백")
            st.write(result["analysis"])
            st.write("### 개선 포인트")
            st.write(result["recommendation"])


# ============================================================
# 6) 실행
# ============================================================

if __name__ == "__main__":
    main()
