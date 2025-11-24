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
# 1. Streamlit 설정
# ============================================================

st.set_page_config(page_title="국민체력100 AI 분석", layout="wide")

# ============================================================
# 2. OpenAI API 설정 (Chat Completions 방식)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"   # Vision 지원

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# 3. KFTA 점수표 (예시값 유지)
# ============================================================

KFTA_SCORES = {
    "situp": {
        "male": {
            "10대": [(55, 100), (50, 90), (45, 80), (40, 70), (35, 60), (30, 50), (25, 40), (20, 30), (15, 20), (10, 10), (0, 0)],
            "20대": [(52, 100), (47, 90), (42, 80), (37, 70), (32, 60), (27, 50), (22, 40), (17, 30), (12, 20), (7, 10), (0, 0)],
            "30대": [(48, 100), (43, 90), (38, 80), (33, 70), (28, 60), (23, 50), (18, 40), (13, 30), (8, 20), (4, 10), (0, 0)],
            "40대": [(44, 100), (39, 90), (34, 80), (29, 70), (24, 60), (19, 50), (14, 40), (9, 30), (5, 20), (2, 10), (0, 0)],
            "50대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30), (4, 20), (2, 10), (0, 0)],
            "60대 이상": [(35, 100), (30, 90), (25, 80), (20, 70), (15, 60), (10, 50), (7, 40), (4, 30), (2, 20), (1, 10), (0, 0)],
        },
        "female": {
            "10대": [(50, 100), (45, 90), (40, 80), (35, 70), (30, 60), (25, 50), (20, 40), (15, 30), (10, 20), (5, 10), (0, 0)],
            "20대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30), (7, 20), (3, 10), (0, 0)],
            "30대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30), (4, 20), (2, 10), (0, 0)],
            "40대": [(36, 100), (31, 90), (26, 80), (21, 70), (16, 60), (11, 50), (8, 40), (5, 30), (3, 20), (1, 10), (0, 0)],
            "50대": [(32, 100), (27, 90), (22, 80), (17, 70), (12, 60), (9, 50), (6, 40), (4, 30), (2, 20), (1, 10), (0, 0)],
            "60대 이상": [(28, 100), (23, 90), (18, 80), (13, 70), (9, 60), (6, 50), (4, 40), (2, 30), (1, 20), (0, 10), (0, 0)],
        },
    },

    "pushup": {
        "male": {
            "10대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30), (5, 20), (2, 10), (0, 0)],
            "20대": [(42, 100), (37, 90), (32, 80), (27, 70), (22, 60), (17, 50), (12, 40), (8, 30), (4, 20), (2, 10), (0, 0)],
            "30대": [(38, 100), (33, 90), (28, 80), (23, 70), (18, 60), (13, 50), (9, 40), (5, 30), (3, 20), (1, 10), (0, 0)],
            "40대": [(34, 100), (29, 90), (24, 80), (19, 70), (14, 60), (10, 50), (7, 40), (4, 30), (2, 20), (1, 10), (0, 0)],
            "50대": [(30, 100), (25, 90), (20, 80), (15, 70), (11, 60), (8, 50), (5, 40), (3, 30), (2, 20), (1, 10), (0, 0)],
            "60대 이상": [(26, 100), (21, 90), (16, 80), (12, 70), (9, 60), (6, 50), (4, 40), (2, 30), (1, 20), (0, 10), (0, 0)],
        },
        "female": {
            "10대": [(35, 100), (30, 90), (25, 80), (20, 70), (16, 60), (12, 50), (8, 40), (5, 30), (3, 20), (1, 10), (0, 0)],
            "20대": [(32, 100), (27, 90), (22, 80), (18, 70), (14, 60), (10, 50), (7, 40), (4, 30), (2, 20), (1, 10), (0, 0)],
            "30대": [(28, 100), (23, 90), (18, 80), (14, 70), (11, 60), (8, 50), (5, 40), (3, 30), (2, 20), (1, 10), (0, 0)],
            "40대": [(24, 100), (19, 90), (15, 80), (11, 70), (8, 60), (6, 50), (4, 40), (2, 30), (1, 20), (0, 10), (0, 0)],
            "50대": [(20, 100), (16, 90), (12, 80), (9, 70), (7, 60), (5, 50), (3, 40), (2, 30), (1, 20), (0, 10), (0, 0)],
            "60대 이상": [(16, 100), (13, 90), (10, 80), (7, 70), (5, 60), (3, 50), (2, 40), (1, 30), (0, 20), (0, 10), (0, 0)],
        },
    },
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
# 4. 프레임 추출 (4장)
# ============================================================

def extract_frames(video_bytes: bytes, num_frames: int = 4):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    duration = count / fps

    idxs = np.linspace(0, count - 1, num_frames, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (640, 360))
            frames.append(f)

    cap.release()
    os.remove(path)
    return frames, duration


# ============================================================
# 5. 이미지 → base64
# ============================================================

def to_b64(img: np.ndarray) -> str:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# 6. OpenAI Vision (ChatCompletions) 호출
# ============================================================

def analyze_with_vlm(frames: List[np.ndarray], duration: float) -> Optional[dict]:
    if not OPENAI_API_KEY:
        return None

    # Vision payload
    image_msgs = []
    for img in frames:
        image_msgs.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{to_b64(img)}"
            }
        })

    system_prompt = """
당신은 국민체력100 분석을 수행하는 AI 코치입니다.
반드시 JSON 형식만 출력하세요.
"""

    user_prompt = f"""
아래는 운동 영상에서 추출된 4개의 프레임입니다.
영상 길이는 {duration:.1f}초입니다.

JSON 형식:

{{
  "exercise_key": "...",
  "exercise_name_kr": "...",
  "estimated_reps": 0,
  "estimated_main_metric": {{"type": "reps|seconds|shuttles", "value": 0}},
  "posture_quality": "...",
  "intensity": "...",
  "stability": "...",
  "risk_flags": [],
  "coach_comment": "..."
}}
"""

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [ {"type": "text", "text": user_prompt}, *image_msgs ]}
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 1200
    }

    for attempt in range(2):
        try:
            res = requests.post(OPENAI_URL, headers=HEADERS, json=payload, timeout=60)
            if res.status_code == 429:
                time.sleep(2)
                continue
            res.raise_for_status()
            data = res.json()
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception:
            if attempt == 1:
                return None
            time.sleep(2)

    return None


# ============================================================
# 7. 점수 계산
# ============================================================

def lookup_kfta(exercise, gender, age_group, value):
    gender_key = "male" if gender == "남성" else "female"

    if exercise in NON_KFTA_EXERCISES or exercise not in KFTA_SCORES:
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

    table = KFTA_SCORES[exercise][gender_key][age_group]

    sc = 0
    for th, s in table:
        if value >= th:
            sc = s
            break

    if sc >= 90: return sc, 1, "매우 우수"
    if sc >= 75: return sc, 2, "우수"
    if sc >= 60: return sc, 3, "보통"
    if sc >= 45: return sc, 4, "주의 필요"
    return sc, 5, "개선 필요"


# ============================================================
# 8. Streamlit UI
# ============================================================

st.title("🏋️‍♂️ AI 기반 국민체력100 자동 분석 (VLM + ChatCompletions)")

col1, col2 = st.columns([1, 2])

with col1:
    age = st.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대", "60대 이상"], index=1)
    gender = st.selectbox("성별", ["남성", "여성"])
    file = st.file_uploader("운동 영상 업로드 (mp4)", type=["mp4"])
    run = st.button("🔍 분석 시작")

with col2:
    if file:
        st.video(file)

st.markdown("---")

if run:
    if not file:
        st.error("영상 파일을 업로드하세요.")
        st.stop()

    # 1) 프레임 추출
    try:
        frames, duration = extract_frames(file.getvalue(), num_frames=4)
        st.success(f"프레임 4장 추출 성공 (영상 길이 {duration:.1f}초)")
    except Exception as e:
        st.error(f"프레임 추출 실패: {e}")
        st.stop()

    # 2) AI 분석
    with st.spinner("AI 분석 중..."):
        result = analyze_with_vlm(frames, duration)

    if not result:
        st.error("AI 분석 실패 (API 문제 or 네트워크 문제)")
        st.stop()

    st.subheader("📌 AI 분석 결과")
    st.json(result)

    # 3) 점수 계산
    metric = result["estimated_main_metric"]
    score, grade, level = lookup_kfta(
        result["exercise_key"],
        gender,
        age,
        float(metric["value"])
    )

    st.subheader("📌 국민체력100 점수")
    st.metric("점수", f"{score} 점")
    st.metric("등급", f"{grade} 등급")
    st.metric("평가", level)
