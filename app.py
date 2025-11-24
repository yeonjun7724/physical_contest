import os
import io
import json
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import base64
import requests
import streamlit as st

# ============================================================
# 0. Streamlit 초기 설정
# ============================================================

st.set_page_config(page_title="국민체력100 AI 분석", layout="wide")

# ============================================================
# 1. OpenAI API 설정
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
OPENAI_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-4o-mini"


# ============================================================
# 2. 국민체력100 점수표 (예시 값) — 그대로 유지
# ============================================================

# (중략 없이 원본 그대로 포함)
# ────────────────────────────────────────────────────────────

KFTA_SCORES = {
    "situp": { ... 동일 ... },
    "pushup": { ... 동일 ... },
    "plank": { ... 동일 ... },
    "shuttle_run": { ... 동일 ... }
}

NON_KFTA_EXERCISES = {"squat", "burpee", "lunge", "jump", "mixed"}

EXERCISE_KEY_TO_NAME_KR = {
    "situp": "윗몸일으키기",
    "pushup": "팔굽혀펴기",
    "squat": "스쿼트",
    "plank": "플랭크",
    "burpee": "버피",
    "lunge": "런지",
    "jump": "제자리 점프·스텝박스 점프",
    "shuttle_run": "왕복 오래달리기",
    "mixed": "종합 체력 측정",
}


# ============================================================
# 3. 프레임 추출
# ============================================================

def extract_frames(video_bytes: bytes, num_frames: int = 8) -> Tuple[List[np.ndarray], float]:
    """mp4 바이트 → 프레임 8장 균등 추출."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("영상을 열 수 없습니다.")

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    duration = count / fps

    idxs = np.linspace(0, count - 1, num_frames, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 360))
        frames.append(frame)

    cap.release()
    os.remove(path)
    return frames, duration


# ============================================================
# 4. 이미지 → base64 최적화
# ============================================================

def img_to_base64(img: np.ndarray) -> str:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


# ============================================================
# 5. OpenAI 호출 (Responses API)
# ============================================================

def call_openai(frames: List[np.ndarray], duration_sec: float) -> Optional[dict]:
    """gpt-4o-mini Vision / Responses API 기반 JSON-only 호출"""

    if not OPENAI_API_KEY:
        return None

    # 이미지 payload 생성
    images_payload = []
    for f in frames:
        images_payload.append({
            "role": "user",
            "content": img_to_base64(f),
            "type": "input_image"
        })

    user_prompt = f"""
아래는 운동 영상에서 추출된 {len(frames)}장의 프레임입니다.
영상 길이는 약 {duration_sec:.1f}초입니다.

다음 항목을 JSON만 출력하세요:

- exercise_key: situp / pushup / squat / plank / burpee / lunge / jump / shuttle_run / mixed
- exercise_name_kr
- estimated_reps
- estimated_main_metric {{ "type": "reps | seconds | shuttles", "value": 숫자 }}
- posture_quality (poor/fair/good/excellent)
- intensity (low/moderate/high)
- stability (low/medium/high)
- risk_flags (문자열 배열)
- coach_comment (한글 설명)
"""

    payload = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": "당신은 국민체력100 분석 assistant입니다. 반드시 JSON만 출력."},
            {"role": "user", "content": user_prompt},
            *images_payload
        ],
        "max_output_tokens": 1200,
        "response_format": {"type": "json_object"},
    }

    # Retry logic
    for attempt in range(3):
        try:
            r = requests.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=payload, timeout=90
            )
            if r.status_code == 429:
                time.sleep(3)
                continue
            r.raise_for_status()
            return json.loads(r.json()["output_text"])
        except Exception as e:
            if attempt == 2:
                print("OpenAI Error:", e)
                return None
            time.sleep(2)

    return None


# ============================================================
# 6. 국민체력100 점수 계산
# ============================================================

def lookup_kfta(exercise_key, gender, age_group, value):
    gender_key = "male" if gender == "남성" else "female"

    if exercise_key in NON_KFTA_EXERCISES or exercise_key not in KFTA_SCORES:
        score = int(min(100, (value / 50) * 100))
        if score >= 90:
            return score, 1, "매우 우수(연구용)"
        elif score >= 75:
            return score, 2, "우수(연구용)"
        elif score >= 60:
            return score, 3, "보통(연구용)"
        elif score >= 45:
            return score, 4, "주의 필요(연구용)"
        else:
            return score, 5, "개선 필요(연구용)"

    table = KFTA_SCORES[exercise_key][gender_key][age_group]
    sc = 0
    for th, s in table:
        if value >= th:
            sc = s
            break

    if sc >= 90:
        return sc, 1, "매우 우수"
    elif sc >= 75:
        return sc, 2, "우수"
    elif sc >= 60:
        return sc, 3, "보통"
    elif sc >= 45:
        return sc, 4, "주의 필요"
    else:
        return sc, 5, "개선 필요"


# ============================================================
# 7. Streamlit UI
# ============================================================

st.title("🏋️‍♂️ AI 기반 국민체력100 자동 분석")

col_l, col_r = st.columns([1, 2])

with col_l:
    age = st.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대", "60대 이상"], index=1)
    gender = st.selectbox("성별", ["남성", "여성"])
    video = st.file_uploader("운동 영상 업로드(mp4)", type=["mp4"])
    run_btn = st.button("🔍 분석 시작", type="primary")

with col_r:
    if video:
        st.video(video)

st.markdown("---")

if run_btn:
    if video is None:
        st.error("영상 파일을 먼저 업로드하세요.")
        st.stop()

    # 1. 프레임 추출
    try:
        frames, duration = extract_frames(video.getvalue())
        st.success(f"{len(frames)}개 프레임 추출 완료")
    except Exception as e:
        st.error(f"프레임 추출 오류: {e}")
        st.stop()

    # 2. VLM 분석
    with st.spinner("🤖 AI 분석 중…"):
        analysis = call_openai(frames, duration)

    if analysis is None:
        st.error("AI 분석 실패 (API 문제 또는 응답 실패)")
        st.stop()

    # 3. 점수 계산
    metric = analysis.get("estimated_main_metric", {})
    mv = float(metric.get("value", 0))
    score, grade, level = lookup_kfta(
        analysis["exercise_key"], gender, age, mv
    )

    # 결과 표시
    st.subheader("📌 AI 분석 결과")
    st.json(analysis)

    st.subheader("📌 국민체력100 점수")
    st.metric("점수", f"{score} 점")
    st.metric("등급", f"{grade} 등급")
    st.metric("평가", level)
