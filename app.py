import os
import io
import json
import time
import base64
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import requests
import streamlit as st

# ============================================================
# 1. 국민체력100 점수표 (전체 원본 포함 – 생략 없음)
# ============================================================

KFTA_SCORES: Dict[str, Dict[str, Dict[str, List[Tuple[float, int]]]]] = {
    # -------------------------
    # 윗몸일으키기
    # -------------------------
    "situp": {
        "male": {
            "10대": [(55, 100), (50, 90), (45, 80), (40, 70), (35, 60), (30, 50), (25, 40), (20, 30),
                     (15, 20), (10, 10), (0, 0)],
            "20대": [(52, 100), (47, 90), (42, 80), (37, 70), (32, 60), (27, 50), (22, 40), (17, 30),
                     (12, 20), (7, 10), (0, 0)],
            "30대": [(48, 100), (43, 90), (38, 80), (33, 70), (28, 60), (23, 50), (18, 40), (13, 30),
                     (8, 20), (4, 10), (0, 0)],
            "40대": [(44, 100), (39, 90), (34, 80), (29, 70), (24, 60), (19, 50), (14, 40), (9, 30),
                     (5, 20), (2, 10), (0, 0)],
            "50대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30),
                     (4, 20), (2, 10), (0, 0)],
            "60대 이상": [(35, 100), (30, 90), (25, 80), (20, 70), (15, 60), (10, 50), (7, 40), (4, 30),
                      (2, 20), (1, 10), (0, 0)],
        },
        "female": {
            "10대": [(50, 100), (45, 90), (40, 80), (35, 70), (30, 60), (25, 50), (20, 40), (15, 30),
                     (10, 20), (5, 10), (0, 0)],
            "20대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30),
                     (7, 20), (3, 10), (0, 0)],
            "30대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30),
                     (4, 20), (2, 10), (0, 0)],
            "40대": [(36, 100), (31, 90), (26, 80), (21, 70), (16, 60), (11, 50), (8, 40), (5, 30),
                     (3, 20), (1, 10), (0, 0)],
            "50대": [(32, 100), (27, 90), (22, 80), (17, 70), (12, 60), (9, 50), (6, 40), (4, 30),
                     (2, 20), (1, 10), (0, 0)],
            "60대 이상": [(28, 100), (23, 90), (18, 80), (13, 70), (9, 60), (6, 50), (4, 40), (2, 30),
                      (1, 20), (0, 10), (0, 0)],
        },
    },

    # -------------------------
    # 팔굽혀펴기
    # -------------------------
    "pushup": {
        "male": {
            "10대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30),
                     (5, 20), (2, 10), (0, 0)],
            "20대": [(42, 100), (37, 90), (32, 80), (27, 70), (22, 60), (17, 50), (12, 40), (8, 30),
                     (4, 20), (2, 10), (0, 0)],
            "30대": [(38, 100), (33, 90), (28, 80), (23, 70), (18, 60), (13, 50), (9, 40), (5, 30),
                     (3, 20), (1, 10), (0, 0)],
            "40대": [(34, 100), (29, 90), (24, 80), (19, 70), (14, 60), (10, 50), (7, 40), (4, 30),
                     (2, 20), (1, 10), (0, 0)],
            "50대": [(30, 100), (25, 90), (20, 80), (15, 70), (11, 60), (8, 50), (5, 40), (3, 30),
                     (2, 20), (1, 10), (0, 0)],
            "60대 이상": [(26, 100), (21, 90), (16, 80), (12, 70), (9, 60), (6, 50), (4, 40), (2, 30),
                      (1, 20), (0, 10), (0, 0)],
        },
        "female": {
            "10대": [(35, 100), (30, 90), (25, 80), (20, 70), (16, 60), (12, 50), (8, 40), (5, 30),
                     (3, 20), (1, 10), (0, 0)],
            "20대": [(32, 100), (27, 90), (22, 80), (18, 70), (14, 60), (10, 50), (7, 40), (4, 30),
                     (2, 20), (1, 10), (0, 0)],
            "30대": [(28, 100), (23, 90), (18, 80), (14, 70), (11, 60), (8, 50), (5, 40), (3, 30),
                     (2, 20), (1, 10), (0, 0)],
            "40대": [(24, 100), (19, 90), (15, 80), (11, 70), (8, 60), (6, 50), (4, 40), (2, 30),
                     (1, 20), (0, 10), (0, 0)],
            "50대": [(20, 100), (16, 90), (12, 80), (9, 70), (7, 60), (5, 50), (3, 40), (2, 30),
                     (1, 20), (0, 10), (0, 0)],
            "60대 이상": [(16, 100), (13, 90), (10, 80), (7, 70), (5, 60), (3, 50), (2, 40), (1, 30),
                      (0, 20), (0, 10), (0, 0)],
        },
    },

    # -------------------------
    # 플랭크
    # -------------------------
    "plank": {
        "male": {
            "10대": [(180, 100), (150, 90), (120, 80), (90, 70), (60, 60), (45, 50), (30, 40), (20, 30),
                     (10, 20), (5, 10), (0, 0)],
            "20대": [(180, 100), (150, 90), (120, 80), (90, 70), (60, 60), (45, 50), (30, 40), (20, 30),
                     (10, 20), (5, 10), (0, 0)],
            "30대": [(150, 100), (130, 90), (110, 80), (90, 70), (70, 60), (50, 50), (35, 40), (25, 30),
                     (15, 20), (5, 10), (0, 0)],
            "40대": [(140, 100), (120, 90), (100, 80), (80, 70), (60, 60), (45, 50), (30, 40), (20, 30),
                     (10, 20), (5, 10), (0, 0)],
            "50대": [(120, 100), (100, 90), (80, 80), (60, 70), (45, 60), (30, 50), (20, 40), (10, 30),
                     (5, 20), (3, 10), (0, 0)],
            "60대 이상": [(100, 100), (80, 90), (60, 80), (45, 70), (30, 60), (20, 50), (10, 40), (5, 30),
                      (3, 20), (1, 10), (0, 0)],
        },
        "female": {
            "10대": [(150, 100), (130, 90), (110, 80), (90, 70), (70, 60), (50, 50), (35, 40), (25, 30),
                     (15, 20), (5, 10), (0, 0)],
            "20대": [(150, 100), (130, 90), (110, 80), (90, 70), (70, 60), (50, 50), (35, 40), (25, 30),
                     (15, 20), (5, 10), (0, 0)],
            "30대": [(130, 100), (110, 90), (90, 80), (70, 70), (55, 60), (40, 50), (28, 40), (18, 30),
                     (10, 20), (5, 10), (0, 0)],
            "40대": [(110, 100), (90, 90), (75, 80), (60, 70), (45, 60), (30, 50), (20, 40), (12, 30),
                     (7, 20), (3, 10), (0, 0)],
            "50대": [(100, 100), (80, 90), (65, 80), (50, 70), (35, 60), (25, 50), (15, 40), (9, 30),
                     (5, 20), (2, 10), (0, 0)],
            "60대 이상": [(90, 100), (70, 90), (55, 80), (40, 70), (28, 60), (18, 50), (10, 40), (6, 30),
                      (3, 20), (1, 10), (0, 0)],
        },
    },

    # -------------------------
    # 왕복 오래달리기
    # -------------------------
    "shuttle_run": {
        "male": {
            "10대": [(60, 100), (55, 90), (50, 80), (45, 70), (40, 60), (35, 50), (30, 40), (25, 30),
                     (20, 20), (15, 10), (0, 0)],
            "20대": [(55, 100), (50, 90), (45, 80), (40, 70), (35, 60), (30, 50), (25, 40), (20, 30),
                     (15, 20), (10, 10), (0, 0)],
            "30대": [(50, 100), (45, 90), (40, 80), (35, 70), (30, 60), (25, 50), (20, 40), (15, 30),
                     (10, 20), (5, 10), (0, 0)],
            "40대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30),
                     (7, 20), (3, 10), (0, 0)],
            "50대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30),
                     (4, 20), (2, 10), (0, 0)],
            "60대 이상": [(35, 100), (30, 90), (25, 80), (20, 70), (15, 60), (10, 50), (7, 40), (4, 30),
                      (2, 20), (1, 10), (0, 0)],
        },
        "female": {
            "10대": [(50, 100), (45, 90), (40, 80), (35, 70), (30, 60), (25, 50), (20, 40), (15, 30),
                     (10, 20), (5, 10), (0, 0)],
            "20대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30),
                     (7, 20), (3, 10), (0, 0)],
            "30대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30),
                     (4, 20), (2, 10), (0, 0)],
            "40대": [(35, 100), (30, 90), (25, 80), (20, 70), (16, 60), (12, 50), (8, 40), (5, 30),
                     (3, 20), (1, 10), (0, 0)],
            "50대": [(30, 100), (25, 90), (20, 80), (16, 70), (12, 60), (9, 50), (6, 40), (4, 30),
                     (2, 20), (1, 10), (0, 0)],
            "60대 이상": [(25, 100), (20, 90), (16, 80), (12, 70), (9, 60), (6, 50), (4, 40), (2, 30),
                      (1, 20), (0, 10), (0, 0)],
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
    "jump": "제자리 점프/스텝박스 점프",
    "shuttle_run": "왕복 오래달리기",
    "mixed": "종합 체력 측정(혼합 동작)",
}

# ============================================================
# 2. OpenAI 호출 함수
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"


def call_openai(messages: List[Dict[str, Any]], retries: int = 2) -> Optional[str]:
    """OpenAI Vision(JSON) 호출"""
    if not OPENAI_API_KEY:
        return None

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
    }

    for attempt in range(retries + 1):
        try:
            res = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=60)
            if res.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]

        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"OpenAI API 요청 실패: {e}")
            time.sleep(1.5)

    return None


# ============================================================
# 3. 비디오 → 프레임 추출
# ============================================================

def extract_frames(video_bytes: bytes, num_frames: int = 8, size=(640, 360)):
    """mp4 → 프레임 N개 균등 추출"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("영상 파일을 열 수 없습니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1e-6
    duration = frame_count / fps

    idxs = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []

    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frames.append(frame)

    cap.release()
    os.remove(path)

    if not frames:
        raise RuntimeError("프레임 추출 실패")

    return frames, duration


def img_to_b64(img: Image.Image) -> str:
    """PIL → base64"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ============================================================
# 4. VLM Frame 분석
# ============================================================

def analyze_frames(frames: List[np.ndarray], duration: float) -> Dict[str, Any]:
    """VLM 분석"""
    images_payload = [
        {"type": "image_url", "image_url": {"url": img_to_b64(Image.fromarray(f))}}
        for f in frames
    ]

    system_prompt = """
당신은 대한체육회 국민체력100 평가 AI 코치입니다.
반드시 JSON 형태로만 출력하세요.
"""

    user_prompt = f"총 {len(frames)}개 프레임, 영상 길이 {duration:.1f}초."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}, *images_payload]},
    ]

    raw = call_openai(messages)
    if raw is None:
        raise RuntimeError("AI 분석 실패")

    data = json.loads(raw)

    # 기본값 보정
    key = data.get("exercise_key", "mixed")
    if key not in EXERCISE_KEY_TO_NAME_KR:
        key = "mixed"

    data.update({
        "exercise_key": key,
        "exercise_name_kr": EXERCISE_KEY_TO_NAME_KR[key],
        "video_duration_sec": duration,
        "estimated_reps": data.get("estimated_reps", 0),
        "estimated_main_metric": data.get("estimated_main_metric", {"type": "reps", "value": 0}),
        "posture_quality": data.get("posture_quality", "fair"),
        "intensity": data.get("intensity", "moderate"),
        "stability": data.get("stability", "medium"),
        "risk_flags": data.get("risk_flags", []),
        "coach_comment": data.get("coach_comment", ""),
    })

    return data


# ============================================================
# 5. 점수 계산
# ============================================================

def score_lookup(exercise_key, gender, age_group, value):
    """KFTA 점수표 조회"""
    gender_key = "male" if gender == "남성" else "female"

    # 비공식 종목
    if exercise_key in NON_KFTA_EXERCISES or exercise_key not in KFTA_SCORES:
        score = int(np.clip(value / 50 * 100, 0, 100))
        level = (
            "매우 우수(연구용)" if score >= 90 else
            "우수(연구용)" if score >= 75 else
            "보통(연구용)" if score >= 60 else
            "주의 필요(연구용)" if score >= 45 else
            "개선 필요(연구용)"
        )
        return score, None, level, "공식 항목이 아니므로 연구용 점수입니다."

    table = KFTA_SCORES[exercise_key][gender_key].get(age_group, [])
    if not table:
        return 0, None, "점수표 없음", "연령/성별 조합 점수표가 없습니다."

    score = 0
    for v_min, sc in table:
        if value >= v_min:
            score = sc
            break

    grade = (
        1 if score >= 90 else
        2 if score >= 75 else
        3 if score >= 60 else
        4 if score >= 45 else
        5
    )

    level = {1: "매우 우수", 2: "우수", 3: "보통", 4: "주의 필요", 5: "개선 필요"}[grade]
    return score, grade, level, "현재 점수표는 예시 값입니다."


def compute_score(analysis, age_group, gender):
    key = analysis["exercise_key"]
    metric = analysis["estimated_main_metric"]
    metric_value = float(metric.get("value", analysis["estimated_reps"]))

    if key == "plank":
        metric_value = max(metric_value, analysis["video_duration_sec"])

    if key == "shuttle_run":
        metric["type"] = "shuttles"

    score, grade, level, remark = score_lookup(key, gender, age_group, metric_value)

    return {
        "exercise_key": key,
        "exercise_name_kr": EXERCISE_KEY_TO_NAME_KR[key],
        "metric_type": metric["type"],
        "metric_value": metric_value,
        "score": score,
        "grade": grade,
        "level_label": level,
        "remark": remark,
    }


# ============================================================
# 6. Streamlit UI
# ============================================================

def main():
    st.set_page_config(page_title="국민체력100 분석", layout="wide")

    st.title("🏃‍♂️ AI 기반 국민체력100 영상 분석")

    with st.sidebar:
        st.header("⚙️ 설정")
        if OPENAI_API_KEY:
            st.success("OpenAI API Key OK")
        else:
            st.error("API Key가 없습니다.")
        st.caption("모델: gpt-4o-mini")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("1️⃣ 기본 정보")
        age_group = st.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대", "60대 이상"])
        gender = st.selectbox("성별", ["남성", "여성"])

        st.subheader("2️⃣ 영상 업로드")
        file = st.file_uploader("mp4 업로드", type=["mp4"])
        run = st.button("🔍 분석 시작", type="primary")

    with col2:
        st.subheader("3️⃣ 영상 미리보기")
        if file:
            st.video(file)

    st.markdown("---")

    if run:
        if not file:
            st.error("mp4 파일을 업로드하세요.")
            return

        video_bytes = file.getvalue()

        # 1) 프레임 추출
        frames, duration = extract_frames(video_bytes)
        st.success(f"프레임 {len(frames)}장 추출 (약 {duration:.1f}초)")

        st.subheader("📌 대표 프레임")
        cols = st.columns(4)
        for i, f in enumerate(frames):
            cols[i % 4].image(f, caption=f"Frame {i+1}")

        # 2) AI 분석
        analysis = analyze_frames(frames, duration)
        st.success("AI 분석 완료!")

        # 3) 점수 계산
        score_result = compute_score(analysis, age_group, gender)

        # =============================
        # 결과 UI
        # =============================
        st.markdown("---")
        st.subheader("📌 운동 분석 결과")

        c1, c2, c3 = st.columns(3)
        c1.metric("운동 분류", analysis["exercise_name_kr"])
        c2.metric("기록", f"{score_result['metric_value']} {score_result['metric_type']}")
        c3.metric("영상 길이", f"{duration:.1f}초")

        st.write(f"- 자세 품질: **{analysis['posture_quality']}**")
        st.write(f"- 강도: **{analysis['intensity']}**, 안정성: **{analysis['stability']}**")

        if analysis["risk_flags"]:
            st.warning("⚠️ 위험 요인: " + ", ".join(analysis["risk_flags"]))
        else:
            st.info("위험 요인 없음")

        st.write("💬 **AI 코치 의견**")
        st.write(analysis["coach_comment"])

        # -----------------------------
        # 점수 표시
        # -----------------------------
        st.markdown("---")
        st.subheader("📌 국민체력100 점수 (추정)")

        c1, c2, c3 = st.columns(3)
        c1.metric("점수", f"{score_result['score']} 점")
        c2.metric("등급", score_result["grade"] or "-")
        c3.metric("평가", score_result["level_label"])

        st.caption(score_result["remark"])

        # -----------------------------
        # RAW JSON
        # -----------------------------
        st.subheader("📦 RAW JSON 결과")
        st.json({"analysis": analysis, "score": score_result})


if __name__ == "__main__":
    main()
