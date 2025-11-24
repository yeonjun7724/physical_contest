import os
import io
import base64
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any

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


# ========= 2. 프레임 추출 (기존 방식) =========

def extract_keyframes(video_path: str, num_frames: int = 8, resize_to=(640, 360)) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("비디오를 열 수 없습니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise RuntimeError("프레임 수를 읽을 수 없습니다.")

    idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resize_to)
        frames.append(frame)

    cap.release()
    return frames


def frames_to_pil(frames: List[np.ndarray]) -> List[Image.Image]:
    return [Image.fromarray(f) for f in frames]


def pil_to_base64(img: Image.Image, format="JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{data}"


# ========= 3A. 프레임 기반 VLM 분석(기존 함수) =========

def analyze_frames_with_vlm(frames: List[Image.Image], duration_sec: float,
                            model="gpt-4.1-mini") -> VLMAnalysisResult:

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # API 없으면 더미값 반환
        return VLMAnalysisResult(
            test_type="squat",
            reps=15,
            duration_sec=duration_sec,
            depth_quality="good",
            knee_alignment="neutral",
            tempo="steady",
            stability="medium",
            risk_flags=[]
        )

    client = OpenAI(api_key=api_key)

    image_contents = []
    for img in frames:
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": pil_to_base64(img)}
        })

    system_prompt = "프레임을 기반으로 스쿼트 동작 분석을 JSON으로만 출력하세요."
    user_prompt = f"영상 길이는 {duration_sec:.1f}초입니다. JSON으로 분석해주세요."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_prompt},
                 *image_contents
             ]}
        ],
        response_format={"type": "json_object"}
    )

    import json
    data = json.loads(resp.choices[0].message.content)

    return VLMAnalysisResult(
        test_type=data.get("test_type", "squat"),
        reps=int(data.get("reps", 0)),
        duration_sec=duration_sec,
        depth_quality=data.get("depth_quality", "fair"),
        knee_alignment=data.get("knee_alignment", "neutral"),
        tempo=data.get("tempo", "steady"),
        stability=data.get("stability", "medium"),
        risk_flags=data.get("risk_flags", [])
    )


# ========= 3B. mp4 직접 VLM 분석(신규 추가) =========

def analyze_video_with_vlm(video_bytes: bytes, duration_sec: float,
                           model="gpt-4.1") -> VLMAnalysisResult:
    """
    업로드한 mp4를 그대로 VLM에게 전달하여 운동 분석.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return VLMAnalysisResult(
            test_type="squat",
            reps=12,
            duration_sec=duration_sec,
            depth_quality="fair",
            knee_alignment="neutral",
            tempo="steady",
            stability="medium",
            risk_flags=[]
        )

    client = OpenAI(api_key=api_key)

    # mp4 → base64 인코딩
    base64_video = base64.b64encode(video_bytes).decode("utf-8")
    video_url = f"data:video/mp4;base64,{base64_video}"

    system_prompt = """
당신은 운동 평가 전문가 AI입니다.
영상을 보고 운동 종류, 반복 수, 자세, 템포, 안정성, 위험요소를 JSON으로만 출력하세요.
"""

    user_prompt = f"영상 길이는 {duration_sec:.1f}초입니다. JSON으로 분석해 주세요."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "input_video", "video_url": video_url}
                ]
            }
        ],
        response_format={"type": "json_object"}
    )

    import json
    data = json.loads(resp.choices[0].message.content)

    return VLMAnalysisResult(
        test_type=data.get("exercise_type", data.get("test_type", "squat")),
        reps=int(data.get("estimated_reps", data.get("reps", 0))),
        duration_sec=duration_sec,
        depth_quality=data.get("movement_quality", {}).get("depth", "fair"),
        knee_alignment=data.get("movement_quality", {}).get("knee_alignment", "neutral"),
        tempo=data.get("tempo", "steady"),
        stability=data.get("stability", "medium"),
        risk_flags=data.get("risk_factors", [])
    )


# ========= 4. 점수화 (기존) =========

def score_against_kfta(result: VLMAnalysisResult,
                       age_group: Optional[str],
                       gender: Optional[str]) -> ScoringResult:
    max_reps = 30
    reps_score = min(result.reps / max_reps * 60, 60)

    depth_map = {"poor": 10, "fair": 20, "good": 30}
    depth_score = depth_map.get(result.depth_quality, 0)

    knee_penalty = 10 if result.knee_alignment == "valgus" else 5 if result.knee_alignment == "varus" else 0

    stability_map = {"low": 0, "medium": 5, "high": 10}
    stability_score = stability_map.get(result.stability, 0)

    tempo_map = {"slow": 5, "steady": 10, "fast": 5}
    tempo_score = tempo_map.get(result.tempo, 0)

    posture_score = depth_score + stability_score + tempo_score - knee_penalty
    posture_score = max(min(posture_score, 40), 0)

    total = int(min(reps_score + posture_score, 100))

    if total >= 90: grade = 1
    elif total >= 75: grade = 2
    elif total >= 60: grade = 3
    elif total >= 45: grade = 4
    else: grade = 5

    is_valid = result.reps >= 5
    detail = "예비측정 가능" if is_valid else "반복 수 부족"

    return ScoringResult(is_valid, total, grade, detail)


# ========= 5. 리포트 생성 (기존) =========

def generate_report_with_llm(analysis: VLMAnalysisResult, score: ScoringResult,
                             model="gpt-4.1-mini") -> str:

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "API 키 없음 - 기본 리포트 사용"

    client = OpenAI(api_key=api_key)

    system_prompt = "당신은 국민체력100 운동 평가 전문가입니다."
    user_prompt = f"""
반복수 {analysis.reps}, 깊이 {analysis.depth_quality}, 무릎 {analysis.knee_alignment},
템포 {analysis.tempo}, 안정성 {analysis.stability}, 총점 {score.total_score}.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return resp.choices[0].message.content


# ========= 6. Streamlit UI =========

def main():
    st.set_page_config(page_title="국민체력100 VLM 데모", layout="centered")
    st.title("AI 기반 국민체력100 영상 분석 (mp4 → VLM)")

    uploaded = st.file_uploader("운동 영상 파일 업로드 (mp4)", type=["mp4"])

    col1, col2 = st.columns(2)
    age_group = col1.selectbox("연령대", ["선택 안 함", "10대", "20대", "30대", "40대", "50대", "60대 이상"])
    gender = col2.selectbox("성별", ["선택 안 함", "남성", "여성"])

    if st.button("분석 실행", type="primary"):
        if uploaded is None:
            st.error("mp4 파일을 업로드하세요.")
            return

        # 영상 저장 (프레임 분석용)
        video_path = "uploaded_video.mp4"
        video_bytes = uploaded.read()

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # 영상 길이
        cap = cv2.VideoCapture(video_path)
        duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1e-6)
        cap.release()

        # 프레임 추출(백업)
        with st.spinner("프레임 추출 중..."):
            frames_np = extract_keyframes(video_path, num_frames=8)

        # mp4 직접 VLM 분석
        with st.spinner("VLM(mp4) 분석 중..."):
            analysis = analyze_video_with_vlm(video_bytes, duration_sec)

        # 점수화
        score = score_against_kfta(
            analysis,
            None if age_group == "선택 안 함" else age_group,
            None if gender == "선택 안 함" else gender
        )

        # 리포트
        with st.spinner("AI 코치 리포트 생성 중..."):
            report = generate_report_with_llm(analysis, score)

        # 출력
        st.subheader("① 추출된 대표 프레임 (참고용)")
        st.image(frames_np, caption=[f"Frame {i+1}" for i in range(len(frames_np))], use_column_width=True)

        st.subheader("② VLM 분석 결과(JSON)")
        st.json(analysis.__dict__)

        st.subheader("③ 점수")
        st.metric("총점", f"{score.total_score}/100")
        st.metric("등급", f"{score.grade} 등급")

        st.subheading = st.subheader("④ AI 코치 리포트")
        st.markdown(report)


if __name__ == "__main__":
    main()
