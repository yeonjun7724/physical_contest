# ============================================
# 국민체력100 AI VLM 종합 분석 시스템
# 완성판 app.py
# ============================================

import cv2
import base64import os
import io
import json
import time
import tempfile
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import requests
import streamlit as st

# ============================================================
# 1. 국민체력100 점수 테이블 (예시) – 실제 표로 교체 가능하게 설계
# ============================================================

"""
실제 국민체력100 공식 점수표를 그대로 옮겨서 아래 딕셔너리에 넣으면 됨.
현재 숫자는 "예시 값"이므로, 반드시 공식 자료 보고 수정해야 함.

구조:
KFTA_SCORES[exercise_key][gender][age_group] = [
    (기준값, 점수),
    (기준값, 점수),
    ...
    (0, 0)
]

- exercise_key: "situp", "pushup", "plank", "shuttle_run" 등
- gender: "male", "female"
- age_group: "10대", "20대", "30대", "40대", "50대", "60대 이상"
- situp/pushup 등: reps(횟수) 기준, plank: seconds(초), shuttle_run: 왕복 횟수 등
"""

KFTA_SCORES: Dict[str, Dict[str, Dict[str, List[Tuple[float, int]]]]] = {
    # 윗몸일으키기 (예시 값)
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
    # 팔굽혀펴기 (예시 값)
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
    # 플랭크 (초 단위, 예시)
    "plank": {
        "male": {
            "10대": [(180, 100), (150, 90), (120, 80), (90, 70), (60, 60), (45, 50), (30, 40), (20, 30), (10, 20), (5, 10), (0, 0)],
            "20대": [(180, 100), (150, 90), (120, 80), (90, 70), (60, 60), (45, 50), (30, 40), (20, 30), (10, 20), (5, 10), (0, 0)],
            "30대": [(150, 100), (130, 90), (110, 80), (90, 70), (70, 60), (50, 50), (35, 40), (25, 30), (15, 20), (5, 10), (0, 0)],
            "40대": [(140, 100), (120, 90), (100, 80), (80, 70), (60, 60), (45, 50), (30, 40), (20, 30), (10, 20), (5, 10), (0, 0)],
            "50대": [(120, 100), (100, 90), (80, 80), (60, 70), (45, 60), (30, 50), (20, 40), (10, 30), (5, 20), (3, 10), (0, 0)],
            "60대 이상": [(100, 100), (80, 90), (60, 80), (45, 70), (30, 60), (20, 50), (10, 40), (5, 30), (3, 20), (1, 10), (0, 0)],
        },
        "female": {
            "10대": [(150, 100), (130, 90), (110, 80), (90, 70), (70, 60), (50, 50), (35, 40), (25, 30), (15, 20), (5, 10), (0, 0)],
            "20대": [(150, 100), (130, 90), (110, 80), (90, 70), (70, 60), (50, 50), (35, 40), (25, 30), (15, 20), (5, 10), (0, 0)],
            "30대": [(130, 100), (110, 90), (90, 80), (70, 70), (55, 60), (40, 50), (28, 40), (18, 30), (10, 20), (5, 10), (0, 0)],
            "40대": [(110, 100), (90, 90), (75, 80), (60, 70), (45, 60), (30, 50), (20, 40), (12, 30), (7, 20), (3, 10), (0, 0)],
            "50대": [(100, 100), (80, 90), (65, 80), (50, 70), (35, 60), (25, 50), (15, 40), (9, 30), (5, 20), (2, 10), (0, 0)],
            "60대 이상": [(90, 100), (70, 90), (55, 80), (40, 70), (28, 60), (18, 50), (10, 40), (6, 30), (3, 20), (1, 10), (0, 0)],
        },
    },
    # 왕복 오래달리기는 측정 방식이 다양해서 여기선 생략/예시만
    "shuttle_run": {
        "male": {
            "10대": [(60, 100), (55, 90), (50, 80), (45, 70), (40, 60), (35, 50), (30, 40), (25, 30), (20, 20), (15, 10), (0, 0)],
            "20대": [(55, 100), (50, 90), (45, 80), (40, 70), (35, 60), (30, 50), (25, 40), (20, 30), (15, 20), (10, 10), (0, 0)],
            "30대": [(50, 100), (45, 90), (40, 80), (35, 70), (30, 60), (25, 50), (20, 40), (15, 30), (10, 20), (5, 10), (0, 0)],
            "40대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30), (7, 20), (3, 10), (0, 0)],
            "50대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30), (4, 20), (2, 10), (0, 0)],
            "60대 이상": [(35, 100), (30, 90), (25, 80), (20, 70), (15, 60), (10, 50), (7, 40), (4, 30), (2, 20), (1, 10), (0, 0)],
        },
        "female": {
            "10대": [(50, 100), (45, 90), (40, 80), (35, 70), (30, 60), (25, 50), (20, 40), (15, 30), (10, 20), (5, 10), (0, 0)],
            "20대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30), (7, 20), (3, 10), (0, 0)],
            "30대": [(40, 100), (35, 90), (30, 80), (25, 70), (20, 60), (15, 50), (10, 40), (7, 30), (4, 20), (2, 10), (0, 0)],
            "40대": [(35, 100), (30, 90), (25, 80), (20, 70), (16, 60), (12, 50), (8, 40), (5, 30), (3, 20), (1, 10), (0, 0)],
            "50대": [(30, 100), (25, 90), (20, 80), (16, 70), (12, 60), (9, 50), (6, 40), (4, 30), (2, 20), (1, 10), (0, 0)],
            "60대 이상": [(25, 100), (20, 90), (16, 80), (12, 70), (9, 60), (6, 50), (4, 40), (2, 30), (1, 20), (0, 10), (0, 0)],
        },
    },
}

# burpee / squat / lunge / jump / mixed 는 공식 항목이 아니라
# "연구용 점수"로 처리 (0~100 정규화)만 해 줄 예정
NON_KFTA_EXERCISES = {"squat", "burpee", "lunge", "jump", "mixed"}


# ============================================================
# 2. OpenAI 호출 유틸 (gpt-4o-mini + Vision, JSON 출력)
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"


def call_openai(messages: List[Dict[str, Any]], max_retries: int = 2) -> Optional[str]:
    """gpt-4o-mini에 Vision + JSON 요청. 실패 시 None 반환 (앱 안 죽게)."""
    if not OPENAI_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "max_tokens": 1200,
    }

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                # rate limit → 짧게 대기 후 재시도
                wait = 3 * (attempt + 1)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            if attempt == max_retries:
                return None
            time.sleep(2 * (attempt + 1))
    return None


# ============================================================
# 3. 비디오 → 프레임 추출
# ============================================================

def extract_frames_from_video_bytes(
    video_bytes: bytes,
    num_frames: int = 8,
    resize_to: Tuple[int, int] = (640, 360),
) -> Tuple[List[np.ndarray], float]:
    """
    mp4 바이트 → 임시파일 → OpenCV로 프레임 균등 추출.
    return: (frames(RGB np.ndarray list), duration_sec)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(tmp_path)
        raise RuntimeError("영상 파일을 열 수 없습니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.000001
    duration_sec = frame_count / fps if frame_count > 0 else 0.0

    if frame_count <= 0:
        cap.release()
        os.remove(tmp_path)
        raise RuntimeError("영상에서 프레임 정보를 읽을 수 없습니다.")

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
    os.remove(tmp_path)

    if not frames:
        raise RuntimeError("프레임을 추출하지 못했습니다.")

    return frames, float(duration_sec)


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


import base64  # pil_to_base64에서 사용


# ============================================================
# 4. 프레임 + 프롬프트 → 운동 분류 & 분석 (gpt-4o-mini)
# ============================================================

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


def analyze_frames_with_vlm(frames: List[np.ndarray], duration_sec: float) -> Dict[str, Any]:
    """
    ① 프레임들을 JPEG → base64로 변환
    ② gpt-4o-mini Vision 모델에 JSON 요청
    ③ 운동 분류 + 반복수 추정 + 자세/위험요인 분석
    """
    images_payload = []
    for f in frames:
        img = Image.fromarray(f)
        b64 = pil_to_base64(img)
        images_payload.append(
            {
                "type": "image_url",
                "image_url": {"url": b64},
            }
        )

    system_prompt = """
당신은 대한체육회 국민체력100 종목 평가를 돕는 AI 코치입니다.
주어진 여러 장의 프레임(운동 영상에서 추출된 이미지)을 보고 다음 운동 중 하나로 분류하세요.

가능한 운동 종류:
- situp: 윗몸일으키기
- pushup: 팔굽혀펴기
- squat: 스쿼트
- plank: 플랭크
- burpee: 버피 테스트/버피 운동
- lunge: 런지(앞/뒤/워킹 포함)
- jump: 제자리 점프 또는 스텝박스 점프
- shuttle_run: 왕복 오래달리기(왕복 달리기, beep test 계열)
- mixed: 여러 동작이 섞여 있어 하나로 분류하기 어려운 경우

반드시 다음 JSON 형식으로만 출력하세요.

{
  "exercise_key": "situp | pushup | squat | plank | burpee | lunge | jump | shuttle_run | mixed",
  "exercise_name_kr": "한글 운동 이름",
  "estimated_reps": 정수 (플랭크/왕복달리기 등도 '반복수' 또는 '왕복 수'로 대략 추정),
  "estimated_main_metric": {
    "type": "reps | seconds | shuttles",
    "value": 숫자
  },
  "posture_quality": "poor | fair | good | excellent",
  "intensity": "low | moderate | high",
  "stability": "low | medium | high",
  "risk_flags": [
    "허리 과신전",
    "무릎 안쪽 모임(내반/외반)",
    "목 긴장",
    "코어 불안정",
    "손목 과사용",
    "호흡 불규칙"
  ] 중 해당하는 것만 선택 (없으면 빈 배열),
  "coach_comment": "짧은 한글 설명으로 운동 특징, 자세 피드백, 주의점 등을 서술"
}
"""

    user_prompt = f"""
아래 이미지는 하나의 운동 영상을 {len(frames)}개 프레임으로 뽑은 것입니다.
실제 영상 길이는 약 {duration_sec:.1f}초 입니다.

- 영상에서 어떤 운동을 하는지 위의 exercise_key 중 하나로 선택
- 반복수, 왕복 수, 버틴 시간 등은 "프레임만 보고 최대한 합리적으로 추정"하세요.
- 국민체력100 공식 종목(윗몸일으키기, 팔굽혀펴기, 오래달리기 등)과 유사한 형태라면 그에 맞게 분류
- JSON 이외 다른 텍스트는 절대 출력하지 마세요.
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt.strip()},
                *images_payload,
            ],
        },
    ]

    raw = call_openai(messages)
    if raw is None:
        raise RuntimeError("AI 분석 호출에 실패했습니다. (API Key, 네트워크, rate limit 확인 필요)")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("AI 응답을 JSON으로 해석할 수 없습니다.")

    # 기본값 보정
    exercise_key = data.get("exercise_key", "mixed")
    if exercise_key not in EXERCISE_KEY_TO_NAME_KR:
        exercise_key = "mixed"
    data["exercise_key"] = exercise_key
    data.setdefault("exercise_name_kr", EXERCISE_KEY_TO_NAME_KR[exercise_key])
    data.setdefault("estimated_reps", 0)
    data.setdefault("estimated_main_metric", {"type": "reps", "value": data.get("estimated_reps", 0)})
    data.setdefault("posture_quality", "fair")
    data.setdefault("intensity", "moderate")
    data.setdefault("stability", "medium")
    data.setdefault("risk_flags", [])
    data.setdefault("coach_comment", "")

    # 실제 영상 길이도 함께 반환 (점수 계산 등에 사용 가능)
    data["video_duration_sec"] = duration_sec

    return data


# ============================================================
# 5. 국민체력100 점수 계산 로직
# ============================================================

def lookup_kfta_score(
    exercise_key: str,
    gender: str,
    age_group: str,
    value: float,
) -> Tuple[int, str, str, str]:
    """
    exercise_key, gender('남성'/'여성'), age_group('20대' 등), 측정값(value)을 기반으로
    국민체력100 점수표에서 점수 찾기.

    return: (score, grade, level_label, remark)
    """
    gender_key = "male" if gender == "남성" else "female"

    # KFTA에 없는 운동은 연구용 점수로 처리
    if exercise_key in NON_KFTA_EXERCISES or exercise_key not in KFTA_SCORES:
        # 0~100 사이로 대략 정규화 (연구용) – 필요시 수정
        # reps/seconds 값이 0~최대 상한(예: 50) 사이로 들어온다고 가정
        max_ref = 50.0
        score = int(max(0, min(100, value / max_ref * 100)))
        # 5등급 분류
        if score >= 90:
            grade, level = 1, "매우 우수(연구용)"
        elif score >= 75:
            grade, level = 2, "우수(연구용)"
        elif score >= 60:
            grade, level = 3, "보통(연구용)"
        elif score >= 45:
            grade, level = 4, "주의 필요(연구용)"
        else:
            grade, level = 5, "개선 필요(연구용)"
        remark = "해당 운동은 국민체력100 공식 기준이 아니므로 연구용 점수로 산정했습니다."
        return score, grade, level, remark

    # 공식 KFTA 항목
    table_exc = KFTA_SCORES.get(exercise_key, {})
    table_gender = table_exc.get(gender_key, {})
    thresholds = table_gender.get(age_group, [])

    if not thresholds:
        # 나이·성별 조합이 테이블에 없을 때
        return 0, 0, "점수표 없음", "해당 연령/성별 조합의 국민체력100 기준표가 등록되어 있지 않습니다."

    # thresholds: [(value_min, score), ...] 순서대로 첫 번째로 만족하는 점수 사용
    score = 0
    for v_min, sc in thresholds:
        if value >= v_min:
            score = sc
            break

    # 점수 → 등급
    if score >= 90:
        grade, level = 1, "매우 우수"
    elif score >= 75:
        grade, level = 2, "우수"
    elif score >= 60:
        grade, level = 3, "보통"
    elif score >= 45:
        grade, level = 4, "주의 필요"
    else:
        grade, level = 5, "개선 필요"

    remark = "점수는 예시 값입니다. 실제 국민체력100 공식 기준값으로 교체해 사용해야 합니다."
    return score, grade, level, remark


def compute_score_from_analysis(
    analysis: Dict[str, Any],
    age_group: str,
    gender: str,
) -> Dict[str, Any]:
    """
    VLM 분석 결과 + 연령/성별 입력 → 국민체력100 점수 계산.
    """
    exercise_key = analysis.get("exercise_key", "mixed")
    metric = analysis.get("estimated_main_metric", {})
    metric_type = metric.get("type", "reps")
    metric_value = float(metric.get("value", analysis.get("estimated_reps", 0)))

    # 플랭크는 seconds, 왕복 달리기는 shuttles 사용
    if exercise_key == "plank":
        metric_type = "seconds"
        metric_value = max(metric_value, analysis.get("video_duration_sec", 0.0))
    elif exercise_key == "shuttle_run":
        metric_type = "shuttles"

    score, grade, level, remark = lookup_kfta_score(
        exercise_key=exercise_key,
        gender=gender,
        age_group=age_group,
        value=metric_value,
    )

    return {
        "exercise_key": exercise_key,
        "exercise_name_kr": EXERCISE_KEY_TO_NAME_KR.get(exercise_key, "알 수 없음"),
        "metric_type": metric_type,
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
    st.set_page_config(page_title="국민체력100 VLM 자동 분석 데모", layout="wide")

    st.title("🏃‍♂️ AI 기반 국민체력100 영상 분석 데모")
    st.markdown(
        """
업로드한 **운동 영상(mp4)**에서 프레임을 추출하고,  
**gpt-4o-mini Vision**을 이용해 운동 종류를 분류하고 국민체력100 기준에 맞춰 점수를 추정하는 데모입니다.

- 지원 운동:
  - 윗몸일으키기, 팔굽혀펴기, 스쿼트, 플랭크, 버피, 런지, 제자리 점프/스텝박스 점프, 왕복 오래달리기, 종합 체력 측정(혼합)
- VLM은 **프레임 기반 추정**이므로 반복수·점수는 실제와 다를 수 있습니다.
- 점수표 숫자는 **예시**이며, 실제 국민체력100 공식 기준값으로 교체할 수 있도록 구조만 맞춰 두었습니다.
"""
    )

    # 사이드바: OpenAI 상태
    with st.sidebar:
        st.header("⚙️ 설정")
        if OPENAI_API_KEY:
            st.success("OpenAI API Key 감지됨 ✅")
        else:
            st.error("OpenAI API Key가 설정되지 않았습니다.\nStreamlit Secrets 또는 환경변수에 `OPENAI_API_KEY`를 넣어주세요.")
        st.markdown("---")
        st.markdown("**모델**: `gpt-4o-mini` (Vision + JSON)")

    # 입력 폼
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("1️⃣ 기본 정보")

        age_group = st.selectbox(
            "연령대",
            ["20대", "10대", "30대", "40대", "50대", "60대 이상"],
            index=1,
            help="국민체력100 기준과 연동될 연령대입니다.",
        )
        gender = st.selectbox(
            "성별",
            ["남성", "여성"],
            index=0,
        )

        st.subheader("2️⃣ 영상 업로드")
        video_file = st.file_uploader(
            "운동 영상 업로드 (mp4 형식)",
            type=["mp4"],
            accept_multiple_files=False,
            help="국민체력100 측정 영상을 촬영한 mp4 파일을 업로드하세요.",
        )

        analyze_button = st.button("🔍 영상 분석 시작", type="primary")

    with col_right:
        st.subheader("3️⃣ 업로드된 영상 미리보기")
        if video_file is not None:
            st.video(video_file)
        else:
            st.info("왼쪽에서 mp4 파일을 업로드하면 여기에서 미리 볼 수 있습니다.")

    st.markdown("---")

    # 분석 실행
    if analyze_button:
        if video_file is None:
            st.error("먼저 mp4 영상 파일을 업로드해주세요.")
            st.stop()

        # ① 바이트 읽기
        video_bytes = video_file.getvalue()

        # ② 프레임 추출
        try:
            with st.spinner("🎞 영상에서 대표 프레임 추출 중..."):
                frames, duration_sec = extract_frames_from_video_bytes(video_bytes, num_frames=8)
        except Exception as e:
            st.error(f"프레임 추출 중 오류가 발생했습니다: {e}")
            st.stop()

        st.success(f"프레임 {len(frames)}장 추출 완료 (영상 길이 약 {duration_sec:.1f}초)")

        # 프레임 미리보기
        st.subheader("4️⃣ 추출된 대표 프레임")
        cols = st.columns(min(len(frames), 4))
        for i, frame in enumerate(frames):
            cols[i % len(cols)].image(frame, caption=f"Frame {i+1}", use_container_width=True)

        # ③ AI 분석
        try:
            with st.spinner("🤖 AI VLM이 운동 종류와 자세를 분석하는 중입니다..."):
                analysis = analyze_frames_with_vlm(frames, duration_sec)
        except Exception as e:
            st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
            st.stop()

        st.success("AI 분석 완료!")

        # ④ 국민체력100 점수 계산
        score_result = compute_score_from_analysis(analysis, age_group=age_group, gender=gender)

        # ⑤ 결과 표시
        st.markdown("---")
        st.subheader("5️⃣ AI 운동 분류 결과")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("운동 분류", f"{analysis['exercise_name_kr']} ({analysis['exercise_key']})")
        with c2:
            st.metric("추정 반복/기록",
                      f"{score_result['metric_value']:.1f} {score_result['metric_type']}")
        with c3:
            st.metric("영상 길이", f"{analysis['video_duration_sec']:.1f} 초")

        st.write("**자세/강도 분석**")
        st.write(
            f"- 자세 품질: **{analysis['posture_quality']}**  "
            f"- 강도: **{analysis['intensity']}**  "
            f"- 안정성: **{analysis['stability']}**"
        )

        if analysis.get("risk_flags"):
            st.warning("⚠️ 위험 요인(추정): " + " / ".join(analysis["risk_flags"]))
        else:
            st.info("특별한 위험 요인이 크게 관찰되지 않았습니다. (VLM 추정)")

        st.markdown("**AI 코치 코멘트**")
        st.write(analysis.get("coach_comment", ""))

        st.markdown("---")
        st.subheader("6️⃣ 국민체력100 기준 점수 (추정)")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("점수 (0~100)", f"{score_result['score']} 점")
        with c2:
            if score_result["grade"] > 0:
                st.metric("등급 (1~5)", f"{score_result['grade']} 등급")
            else:
                st.metric("등급 (1~5)", "기준 없음")
        with c3:
            st.metric("평가", score_result["level_label"])

        st.caption(score_result["remark"])

        st.markdown("---")
        st.subheader("7️⃣ 원본 AI JSON 결과 (디버깅/연구용)")
        col_json1, col_json2 = st.columns(2)
        with col_json1:
            st.write("🔎 VLM 분석 Raw JSON")
            st.json(analysis)
        with col_json2:
            st.write("🧮 점수 계산 결과 JSON")
            st.json(score_result)


# ============================================================
# 7. 실행
# ============================================================

if __name__ == "__main__":
    main()

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
