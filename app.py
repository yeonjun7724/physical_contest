import os
import io
import base64
from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any

import cv2
import numpy as np
from pytube import YouTube
from PIL import Image

import streamlit as st

# OpenAI (llm/vlm) 사용 시
from openai import OpenAI


# ========= 1. 데이터 구조 =========

@dataclass
class VLMAnalysisResult:
    test_type: str               # ex) "squat"
    reps: int                    # 반복 횟수
    duration_sec: float          # 전체 영상 길이
    depth_quality: Literal["poor", "fair", "good"]
    knee_alignment: Literal["valgus", "varus", "neutral"]
    tempo: Literal["slow", "steady", "fast"]
    stability: Literal["low", "medium", "high"]
    risk_flags: List[str]        # ["knee_in", "lumbar_hyperextension", ...]


@dataclass
class ScoringResult:
    is_valid_for_kfta: bool      # 국민체력100 예비측정으로 쓸 수 있는지
    total_score: int             # 0~100
    grade: int                   # 1~5 (1이 가장 좋음)
    detail: str                  # 텍스트 설명


# ========= 2. 유튜브 유틸 =========

def normalize_youtube_url(url: str) -> str:
    """Shorts 링크를 watch 링크로 변환 (간단 버전)."""
    url = url.strip()
    if "youtube.com/shorts/" in url:
        vid = url.split("shorts/")[-1].split("?")[0]
        return f"https://www.youtube.com/watch?v={vid}"
    return url


def download_youtube_video(url: str, out_dir: str = "downloads") -> str:
    """유튜브 영상을 mp4로 다운로드하고 파일 경로를 반환."""
    os.makedirs(out_dir, exist_ok=True)
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension="mp4", progressive=True).get_highest_resolution()
    file_path = stream.download(output_path=out_dir)
    return file_path


# ========= 3. 프레임 추출 =========

def extract_keyframes(
    video_path: str,
    num_frames: int = 8,
    resize_to: tuple[int, int] = (640, 360)
) -> List[np.ndarray]:
    """영상에서 일정 간격으로 num_frames 장의 프레임 추출."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("비디오를 열 수 없습니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise RuntimeError("프레임 수를 읽을 수 없습니다.")

    # 균등 간격 인덱스
    idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    frames: List[np.ndarray] = []
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


def pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


# ========= 4. LLM/VLM 프레임 분석 =========

def analyze_frames_with_vlm(
    frames: List[Image.Image],
    duration_sec: float,
    model: str = "gpt-4.1-mini"  # 또는 gpt-4o-mini 등 멀티모달 지원 모델
) -> VLMAnalysisResult:
    """
    여러 프레임을 LLM/VLM에 넣어서 스쿼트 동작을 분석.
    - 실제로는 OpenAI 멀티모달 API를 호출
    - 여기서는 JSON 형식으로 응답하도록 프롬프트 설계
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # API 키 없으면 더미 결과 반환 (데모용)
        return VLMAnalysisResult(
            test_type="squat",
            reps=18,
            duration_sec=duration_sec,
            depth_quality="good",
            knee_alignment="neutral",
            tempo="steady",
            stability="medium",
            risk_flags=[]
        )

    client = OpenAI(api_key=api_key)

    # 이미지들을 data URL로 변환
    image_contents: List[Dict[str, Any]] = []
    for img in frames:
        data_url = pil_to_base64(img, format="JPEG")
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": data_url
            }
        })

    system_prompt = """
당신은 국민체력100 스쿼트 종목 평가를 돕는 전문가 코치입니다.
사용자로부터 스쿼트 운동 영상을 대표하는 여러 프레임 이미지를 받습니다.
이 이미지를 보고 아래 항목을 JSON으로만 출력하세요.

출력 JSON 필드:
- test_type: "squat" 또는 다른 종목 이름 (영문 소문자)
- reps: 대략적인 반복 횟수 (정수)
- depth_quality: "poor" | "fair" | "good"
- knee_alignment: "valgus" | "varus" | "neutral"
- tempo: "slow" | "steady" | "fast"
- stability: "low" | "medium" | "high"
- risk_flags: 리스트. 예) ["knee_in", "lumbar_hyperextension"] 없으면 [].

다른 설명 텍스트는 절대 쓰지 말고, JSON 객체만 반환하세요.
"""

    user_prompt = f"""
다음 이미지는 한 사람의 스쿼트 동작을 촬영한 영상에서 추출한 프레임입니다.
영상 길이는 약 {duration_sec:.1f}초입니다.
프레임들을 보고 위에서 설명한 JSON 형식에 맞게 평가해주세요.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *image_contents
            ]
        }
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"}
    )

    content = resp.choices[0].message.content
    import json
    data = json.loads(content)

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


# ========= 5. 국민체력100 점수화 (데모) =========

def score_against_kfta(
    result: VLMAnalysisResult,
    age_group: Optional[str] = None,
    gender: Optional[str] = None
) -> ScoringResult:
    """
    실제 국민체력100 공식 등급표가 아니라,
    데모용 로직. 나중에 공식 테이블로 치환하면 됨.
    """

    # 1) 반복수 점수 (예: 30회 이상이면 최대 60점)
    max_reps = 30
    reps_score = min(result.reps / max_reps * 60, 60)

    # 2) 자세 점수
    depth_map = {"poor": 10, "fair": 20, "good": 30}
    depth_score = depth_map.get(result.depth_quality, 0)

    knee_penalty = 0
    if result.knee_alignment == "valgus":
        knee_penalty = 10
    elif result.knee_alignment == "varus":
        knee_penalty = 5

    stability_map = {"low": 0, "medium": 5, "high": 10}
    stability_score = stability_map.get(result.stability, 0)

    tempo_map = {"slow": 5, "steady": 10, "fast": 5}
    tempo_score = tempo_map.get(result.tempo, 0)

    posture_score = depth_score + stability_score + tempo_score - knee_penalty
    posture_score = max(min(posture_score, 40), 0)

    total = int(reps_score + posture_score)
    total = max(min(total, 100), 0)

    # 등급(데모)
    if total >= 90:
        grade = 1
    elif total >= 75:
        grade = 2
    elif total >= 60:
        grade = 3
    elif total >= 45:
        grade = 4
    else:
        grade = 5

    # 예비측정 적합 여부(데모)
    is_valid = True
    reason: List[str] = []

    if result.test_type != "squat":
        is_valid = False
        reason.append("스쿼트 종목 영상이 아닙니다.")
    if result.reps < 5:
        is_valid = False
        reason.append("반복 횟수가 너무 적어 예비측정으로 사용하기 어렵습니다.")
    if result.knee_alignment == "valgus":
        reason.append("무릎이 안쪽으로 심하게 모여 부상 위험이 있습니다.")
    if "lumbar_hyperextension" in result.risk_flags:
        reason.append("허리 과신전(요추 과도한 꺾임) 위험이 있습니다.")

    detail_text = " / ".join(reason) if reason else "국민체력100 스쿼트 예비측정으로 활용 가능한 영상입니다. (데모 기준)"

    return ScoringResult(
        is_valid_for_kfta=is_valid,
        total_score=total,
        grade=grade,
        detail=detail_text
    )


# ========= 6. LLM 기반 코치 레포트 =========

def generate_report_with_llm(
    analysis: VLMAnalysisResult,
    score: ScoringResult,
    model: str = "gpt-4.1-mini"
) -> str:
    """
    분석 결과 + 점수 기반으로 LLM이 한국어 리포트 작성.
    - OPENAI_API_KEY 없으면 규칙 기반 텍스트로 대체.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # LLM 사용 불가 시 간단한 규칙 기반 리포트
        lines = []
        lines.append("### 1. 종합 평가 요약")
        lines.append(
            f"- 이 영상은 스쿼트 {analysis.reps}회를 수행한 기록으로, "
            f"데모 기준 총점 {score.total_score}점, 예상 {score.grade}등급 수준으로 평가되었습니다."
        )

        lines.append("\n### 2. 잘 수행한 점")
        if analysis.depth_quality == "good":
            lines.append("- 스쿼트 깊이가 충분하여 하체 근력을 잘 활용하고 있습니다.")
        if analysis.knee_alignment == "neutral":
            lines.append("- 무릎 정렬이 비교적 안정적이며, 큰 부상 위험 신호는 보이지 않습니다.")
        if analysis.tempo == "steady":
            lines.append("- 반복 속도가 일정하여 측정 및 훈련에 적합한 템포를 유지하고 있습니다.")

        lines.append("\n### 3. 개선이 필요한 부분")
        if analysis.stability != "high":
            lines.append("- 상체와 하체의 안정성을 조금 더 높일 필요가 있습니다. 코어 근력과 균형 훈련을 병행하면 좋습니다.")
        if analysis.knee_alignment == "valgus":
            lines.append("- 스쿼트 시 무릎이 안쪽으로 모이지 않도록 엉덩이와 둔근 활성화에 신경 써야 합니다.")
        if score.total_score < 60:
            lines.append("- 반복 횟수와 자세 완성도를 조금씩 끌어올려 국민체력100 상위 등급을 노려볼 수 있습니다.")

        lines.append("\n### 4. 다음 단계 추천 운동")
        lines.append("- 발목 및 고관절 가동성 향상 스트레칭 5분")
        lines.append("- 벽 스쿼트 또는 의자를 활용한 박스 스쿼트로 올바른 자세 연습 5~10분")
        lines.append("- 플랭크, 데드버그 등 코어 안정화 운동 5분")

        if not score.is_valid_for_kfta:
            lines.append(
                "\n※ 현재 영상은 데모 기준으로 국민체력100 예비측정용으로는 일부 한계가 있습니다. "
                "공식 인증을 위해서는 국민체력100 센터 또는 공식 가이드에 맞춘 촬영 환경을 권장합니다."
            )

        return "\n".join(lines)

    # LLM 사용 가능하면 OpenAI 호출
    client = OpenAI(api_key=api_key)

    system_prompt = """
당신은 대한민국 국민체력100 프로그램을 잘 이해하고 있는 AI 체력 코치입니다.
사용자의 스쿼트 영상 분석 결과를 바탕으로,
친절하고 구체적인 한국어 리포트를 작성합니다.
"""

    user_prompt = f"""
다음은 사용자의 스쿼트 영상에 대한 VLM 분석 결과와 국민체력100 예비측정 점수입니다.

[분석 결과]
- 종목: {analysis.test_type}
- 반복 횟수: {analysis.reps}회
- 영상 길이: {analysis.duration_sec:.1f}초
- 스쿼트 깊이: {analysis.depth_quality}
- 무릎 정렬: {analysis.knee_alignment}
- 템포: {analysis.tempo}
- 안정성: {analysis.stability}
- 위험 신호: {', '.join(analysis.risk_flags) if analysis.risk_flags else '없음'}

[점수화 결과]
- 총점: {score.total_score} / 100
- 예상 등급: {score.grade}등급 (1등급이 가장 우수)
- 예비측정 적합 여부: {"적합" if score.is_valid_for_kfta else "부적합"}
- 설명: {score.detail}

위 정보를 바탕으로, 다음 형식에 맞추어 리포트를 작성해 주세요.

1. 종합 평가 요약 (2~3문장)
2. 잘 수행한 점 (3~4문장)
3. 개선이 필요한 부분 (자세, 안정성, 속도, 위험요소 중심으로 4~6문장)
4. 국민체력100 센터 방문 전, 집에서 연습하면 좋은 운동 2~3가지와 간단 설명
5. 예비측정으로 활용할 때 주의해야 할 점 (필요 시)

문장은 모두 한국어로 작성해 주세요.
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return resp.choices[0].message.content


# ========= 7. Streamlit UI =========

def main():
    st.set_page_config(page_title="국민체력100 VLM 데모", layout="centered")
    st.title("AI 기반 국민체력100 영상 분석 데모")
    st.write(
        "유튜브 운동 영상 링크를 입력하면, "
        "프레임을 추출해 VLM/LLM이 국민체력100 예비측정 기준으로 분석하고 "
        "점수와 코치 리포트를 생성합니다. (현재는 데모 로직)"
    )

    url = st.text_input(
        "유튜브 영상 링크를 입력하세요",
        placeholder="https://www.youtube.com/shorts/4Bc1tPaYkOo"
    )

    col1, col2 = st.columns(2)
    with col1:
        age_group = st.selectbox("연령대 (선택)", ["선택 안 함", "10대", "20대", "30대", "40대", "50대", "60대 이상"])
    with col2:
        gender = st.selectbox("성별 (선택)", ["선택 안 함", "남성", "여성", "기타"])

    if st.button("분석 실행", type="primary"):
        if not url:
            st.error("유튜브 링크를 입력해 주세요.")
            return

        norm_url = normalize_youtube_url(url)
        st.info(f"처리할 유튜브 링크: {norm_url}")

        # 1) 유튜브 다운로드
        with st.spinner("유튜브 영상 다운로드 중..."):
            try:
                video_path = download_youtube_video(norm_url)
            except Exception as e:
                st.error(f"영상 다운로드 실패: {e}")
                return

        # 2) 프레임 추출
        with st.spinner("프레임 추출 중..."):
            try:
                cap = cv2.VideoCapture(video_path)
                duration_sec = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(
                    cap.get(cv2.CAP_PROP_FPS), 1e-6
                )
                cap.release()

                frames_np = extract_keyframes(video_path, num_frames=8, resize_to=(640, 360))
                frames_pil = frames_to_pil(frames_np)
            except Exception as e:
                st.error(f"프레임 추출 실패: {e}")
                return

        # 3) VLM 분석
        with st.spinner("VLM/LLM 기반 자세 분석 중..."):
            try:
                analysis = analyze_frames_with_vlm(frames_pil, duration_sec)
            except Exception as e:
                st.error(f"VLM 분석 실패: {e}")
                return

        # 4) 점수화
        score = score_against_kfta(
            analysis,
            age_group=None if age_group == "선택 안 함" else age_group,
            gender=None if gender == "선택 안 함" else gender,
        )

        # 5) 레포트 생성
        with st.spinner("AI 코치 리포트 생성 중..."):
            try:
                report = generate_report_with_llm(analysis, score)
            except Exception as e:
                st.error(f"리포트 생성 실패: {e}")
                return

        # ===== 결과 출력 =====
        st.subheader("1. 추출된 핵심 프레임")
        st.caption("VLM이 참고한 대표 프레임들입니다.")
        st.image(frames_np, caption=[f"Frame {i+1}" for i in range(len(frames_np))], use_column_width=True)

        st.subheader("2. 국민체력100 예비측정 적합 여부 (데모)")
        if score.is_valid_for_kfta:
            st.success("예비측정 기준에 사용할 수 있는 영상으로 판단됩니다. (데모 기준)")
        else:
            st.warning("예비측정 기준에 일부 부적합한 영상으로 판단됩니다. (데모 기준)")
        st.write(score.detail)

        st.subheader("3. 점수 및 예상 등급 (데모)")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("총점", f"{score.total_score} / 100")
        with col_b:
            st.metric("예상 등급", f"{score.grade} 등급")

        st.subheader("4. VLM 분석 요약 (데모)")
        st.json({
            "종목": analysis.test_type,
            "반복 횟수": analysis.reps,
            "영상 길이(초)": round(analysis.duration_sec, 1),
            "스쿼트 깊이": analysis.depth_quality,
            "무릎 정렬": analysis.knee_alignment,
            "템포": analysis.tempo,
            "안정성": analysis.stability,
            "위험 신호": analysis.risk_flags,
        })

        st.subheader("5. AI 코치 레포트")
        st.markdown(report)


if __name__ == "__main__":
    main()
