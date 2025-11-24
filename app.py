import os
import io
import base64
from dataclasses import dataclass
from typing import List, Optional, Literal

import cv2
import numpy as np
from PIL import Image

import streamlit as st
from openai import OpenAI


# ========= 1. 데이터 구조 =========

@dataclass
class VLMAnalysisResult:
    test_type: str
    reps: int
    duration_sec: float
    depth_quality: Literal["poor", "fair", "good"]
    knee_alignment: Literal["valgus", "varus", "neutral"]
    tempo: Literal["slow", "steady", "fast"]
    stability: Literal["low", "medium", "high"]
    risk_flags: List[str]


@dataclass
class ScoringResult:
    is_valid_for_kfta: bool
    total_score: int
    grade: int
    detail: str


# ========= 2. 프레임 추출 =========

def extract_keyframes(video_path: str, num_frames: int = 8,
                      resize_to=(640, 360)) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, resize_to)
            frames.append(frame)
    cap.release()
    return frames


# ========= 3. mp4 영상 → VLM 분석 =========

def analyze_video_with_vlm(video_bytes, duration_sec, model="gpt-4.1"):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    # mp4 → base64
    b64 = base64.b64encode(video_bytes).decode()
    video_url = f"data:video/mp4;base64,{b64}"

    system_prompt = """
당신은 운동 분석 전문가입니다.
영상을 보고 운동 분석 결과를 JSON으로만 출력하세요.
"""

    user_prompt = f"영상 길이는 {duration_sec:.1f}초입니다. JSON으로 분석해 주세요."

    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "input_video", "video_url": video_url},
                ],
            },
        ]
    )

    import json
    data = json.loads(resp.choices[0].message.content)

    return VLMAnalysisResult(
        test_type=data.get("exercise_type", "squat"),
        reps=int(data.get("estimated_reps", 0)),
        duration_sec=duration_sec,
        depth_quality=data.get("movement_quality", {}).get("depth", "fair"),
        knee_alignment=data.get("movement_quality", {}).get("knee_alignment", "neutral"),
        tempo=data.get("tempo", "steady"),
        stability=data.get("stability", "medium"),
        risk_flags=data.get("risk_factors", []),
    )


# ========= 4. 점수화 =========

def score_against_kfta(a: VLMAnalysisResult, age_group, gender):
    max_reps = 30
    reps_score = min(a.reps / max_reps * 60, 60)

    depth_map = {"poor": 10, "fair": 20, "good": 30}
    depth_score = depth_map.get(a.depth_quality, 0)

    tempo_map = {"slow": 5, "steady": 10, "fast": 5}
    tempo_score = tempo_map.get(a.tempo, 0)

    stability_map = {"low": 0, "medium": 5, "high": 10}
    stability_score = stability_map.get(a.stability, 0)

    knee_penalty = 10 if a.knee_alignment == "valgus" else 5 if a.knee_alignment == "varus" else 0

    posture_score = depth_score + stability_score + tempo_score - knee_penalty
    posture_score = max(min(posture_score, 40), 0)

    total = int(min(reps_score + posture_score, 100))

    grade = 1 if total >= 90 else 2 if total >= 75 else 3 if total >= 60 else 4 if total >= 45 else 5

    is_valid = a.reps >= 5
    detail = "예비측정 가능" if is_valid else "반복 수 부족"

    return ScoringResult(is_valid, total, grade, detail)


# ========= 5. 리포트 =========

def generate_report_with_llm(analysis, score, model="gpt-4.1-mini"):
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    system_prompt = "운동 분석 전문가입니다. 아래 정보를 바탕으로 상세 리포트를 작성하세요."

    user_prompt = f"""
반복수: {analysis.reps}
스쿼트 깊이: {analysis.depth_quality}
무릎 정렬: {analysis.knee_alignment}
템포: {analysis.tempo}
안정성: {analysis.stability}
총점: {score.total_score}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return resp.choices[0].message.content


# ========= 6. Streamlit UI (새 UX/UI) =========

def main():

    st.set_page_config(page_title="국민체력100 AI 영상 분석", layout="centered")

    st.title("AI 기반 국민체력100 영상 분석 (mp4 업로드 전용)")
    st.write("업로드한 운동 영상을 기반으로 VLM이 자동으로 스쿼트 자세를 분석합니다.")

    # 🔥🔥🔥 유튜브 링크 제거 → 업로드 컴포넌트 추가 🔥🔥🔥
    uploaded = st.file_uploader("운동 영상 업로드 (mp4)", type=["mp4"])

    # 업로드한 영상 미리보기
    if uploaded:
        st.video(uploaded)

    col1, col2 = st.columns(2)
    age_group = col1.selectbox("연령대", ["선택 안 함", "10대", "20대", "30대", "40대", "50대", "60대 이상"])
    gender = col2.selectbox("성별", ["선택 안 함", "남성", "여성"])

    if st.button("분석 실행", type="primary"):

        if uploaded is None:
            st.error("mp4 파일을 업로드해주세요.")
            return

        video_bytes = uploaded.read()

        # ---- 영상 길이 ----
        video_path = "uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(video_path)
        fps = max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
        duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()

        # ---- 대표 프레임 추출 ----
        with st.spinner("대표 프레임 추출 중..."):
            frames_np = extract_keyframes(video_path)

        # ---- mp4 영상 직접 분석 ----
        with st.spinner("VLM이 영상을 분석하는 중..."):
            analysis = analyze_video_with_vlm(video_bytes, duration_sec)

        # ---- 점수화 ----
        score = score_against_kfta(analysis,
                                   None if age_group == "선택 안 함" else age_group,
                                   None if gender == "선택 안 함" else gender)

        # ---- 리포트 생성 ----
        with st.spinner("AI 리포트 생성 중..."):
            report = generate_report_with_llm(analysis, score)

        # 결과 출력
        st.subheader("1. 대표 프레임")
        st.image(frames_np, caption=[f"Frame {i+1}" for i in range(len(frames_np))], use_column_width=True)

        st.subheader("2. VLM 분석 결과(JSON)")
        st.json(analysis.__dict__)

        st.subheader("3. 점수 결과")
        st.metric("총점", f"{score.total_score} / 100")
        st.metric("예상 등급", f"{score.grade} 등급")

        st.subheader("4. AI 코치 리포트")
        st.markdown(report)


if __name__ == "__main__":
    main()
