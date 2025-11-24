import os
import io
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import streamlit as st
import mediapipe as mp

# ============================================================
# 1. Streamlit 기본 설정
# ============================================================

st.set_page_config(page_title="국민체력100 운동 분석 (MediaPipe 버전)", layout="wide")

st.title("🏋️‍♂️ AI 없이도 동작하는 국민체력100 운동 분석 웹앱 (MediaPipe Pose)")

st.markdown(
    """
이 웹앱은 **OpenAI / 클라우드 API 없이** 로컬에서 실행 가능한 **MediaPipe Pose** 기반 운동 분석 도구입니다.  

- 업로드한 **운동 영상(mp4)** 을 프레임 단위로 읽어,
- **MediaPipe Pose** 로 관절을 추출하고,
- 운동 종류(푸시업/스쿼트/윗몸일으키기/플랭크/그 외)를 **간단한 규칙 기반으로 분류**하고,
- **반복 횟수(또는 플랭크 유지 시간)를 추정**한 뒤,
- 기본적인 **국민체력100 예시 점수표**를 활용해 점수를 계산합니다.

> 실제 평가에 쓰기 전에는 반드시 충분한 테스트와 보정이 필요합니다. (연구/프로토타입 용도)
"""
)

# ============================================================
# 2. MediaPipe Pose 설정
# ============================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ============================================================
# 3. KFTA 점수표 (예시값 – 연준이 쓰던 구조 그대로)
# ============================================================

KFTA_SCORES: Dict[str, Dict[str, Dict[str, List[Tuple[float, int]]]]] = {
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
}

# shuttle_run은 예시만 (지금 MediaPipe로 자동 인식하진 않지만 구조 유지)
KFTA_SCORES["shuttle_run"] = {
    "male": {
        "10대": [(60, 100), (55, 90), (50, 80), (45, 70), (40, 60), (35, 50), (30, 40), (25, 30), (20, 20), (15, 10), (0, 0)],
        "20대": [(55, 100), (50, 90), (45, 80), (40, 70), (35, 60), (30, 50), (25, 40), (20, 30), (15, 20), (10, 10), (0, 0)],
    },
    "female": {
        "10대": [(50, 100), (45, 90), (40, 80), (35, 70), (30, 60), (25, 50), (20, 40), (15, 30), (10, 20), (5, 10), (0, 0)],
        "20대": [(45, 100), (40, 90), (35, 80), (30, 70), (25, 60), (20, 50), (15, 40), (10, 30), (7, 20), (3, 10), (0, 0)],
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
    "mixed": "혼합/기타",
}

# ============================================================
# 4. 기하학 유틸
# ============================================================

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 벡터 사이 각도 (deg)"""
    v1 = v1.astype(float)
    v2 = v2.astype(float)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """관절 B 기준 각도 (A-B-C, deg)"""
    return angle_between(a - b, c - b)


def moving_average(arr: np.ndarray, window: int = 5) -> np.ndarray:
    if len(arr) < window:
        return arr
    cumsum = np.cumsum(np.insert(arr, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / float(window)


def count_reps_from_series(series: np.ndarray, low_th: float, high_th: float) -> int:
    """
    시계열 데이터를 기준으로 up↔down 사이클 수를 세는 간단한 카운터.
    - series: 각도 또는 위치 시계열
    - low_th: 'down' 기준 값
    - high_th: 'up' 기준 값
    """
    if len(series) == 0:
        return 0
    state = "up"
    reps = 0
    for v in series:
        if state == "up" and v < low_th:
            state = "down"
        elif state == "down" and v > high_th:
            state = "up"
            reps += 1
    return reps

# ============================================================
# 5. 비디오 분석 (MediaPipe Pose)
# ============================================================

def analyze_video_with_mediapipe(video_bytes: bytes) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    """
    mp4 바이트 → MediaPipe Pose 분석 → 운동 분류 + 반복수 추정
    return:
        analysis_dict, preview_frames(list of RGB np.ndarray)
    """
    # 임시 파일 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(tmp_path)
        raise RuntimeError("영상 파일을 열 수 없습니다.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    duration = frame_count / fps if frame_count > 0 else 0.0

    # 분석용 샘플링 간격 (약 3fps 수준)
    step = max(1, int(round(fps / 3)))
    preview_step = max(1, frame_count // 4) if frame_count > 0 else step

    torso_angles = []
    knee_angles = []
    elbow_angles = []
    hip_heights = []

    preview_frames: List[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프리뷰용 이미지 저장 (4장 정도)
            if idx % preview_step == 0:
                rgb_small = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_small = cv2.resize(rgb_small, (640, 360))
                preview_frames.append(rgb_small)

            # 분석용 샘플링
            if idx % step == 0:
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark

                    def get_xy(name: int) -> np.ndarray:
                        return np.array([lm[name].x * w, lm[name].y * h])

                    # 주요 관절 좌표
                    ls = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER)
                    rs = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER)
                    lh = get_xy(mp_pose.PoseLandmark.LEFT_HIP)
                    rh = get_xy(mp_pose.PoseLandmark.RIGHT_HIP)
                    lk = get_xy(mp_pose.PoseLandmark.LEFT_KNEE)
                    rk = get_xy(mp_pose.PoseLandmark.RIGHT_KNEE)
                    la = get_xy(mp_pose.PoseLandmark.LEFT_ANKLE)
                    ra = get_xy(mp_pose.PoseLandmark.RIGHT_ANKLE)
                    le = get_xy(mp_pose.PoseLandmark.LEFT_ELBOW)
                    re = get_xy(mp_pose.PoseLandmark.RIGHT_ELBOW)
                    lw = get_xy(mp_pose.PoseLandmark.LEFT_WRIST)
                    rw = get_xy(mp_pose.PoseLandmark.RIGHT_WRIST)

                    shoulder = (ls + rs) / 2.0
                    hip = (lh + rh) / 2.0
                    knee = (lk + rk) / 2.0
                    ankle = (la + ra) / 2.0
                    elbow = (le + re) / 2.0
                    wrist = (lw + rw) / 2.0

                    # 몸통 각도 (0 = 수직, 90 = 수평에 가까움)
                    torso_vec = shoulder - hip
                    vertical_vec = np.array([0, -1])
                    torso_angle = angle_between(torso_vec, vertical_vec)
                    torso_angles.append(torso_angle)

                    # 무릎 각도 (스쿼트/런지/PJ 등에 사용)
                    knee_angle = joint_angle(hip, knee, ankle)
                    knee_angles.append(knee_angle)

                    # 팔꿈치 각도 (푸시업 등에 사용)
                    elbow_angle = joint_angle(shoulder, elbow, wrist)
                    elbow_angles.append(elbow_angle)

                    # 엉덩이 높이 (정규화)
                    hip_height = hip[1] / h
                    hip_heights.append(hip_height)

            idx += 1

    cap.release()
    os.remove(tmp_path)

    torso_arr = np.array(torso_angles)
    knee_arr = np.array(knee_angles)
    elbow_arr = np.array(elbow_angles)
    hip_arr = np.array(hip_heights)

    if len(preview_frames) == 0:
        raise RuntimeError("포즈를 인식할 수 있는 프레임이 없습니다.")

    if len(torso_arr) == 0:
        # 포즈 추출 실패
        analysis = {
            "exercise_key": "mixed",
            "exercise_name_kr": "혼합/인식 불가",
            "reps": 0,
            "metric_type": "reps",
            "metric_value": 0,
            "posture_quality": "unknown",
            "intensity": "unknown",
            "stability": "unknown",
            "notes": "사람 포즈를 안정적으로 인식하지 못했습니다. 카메라 구도/조명/거리 등을 조정해 주세요.",
            "video_duration_sec": duration,
        }
        return analysis, preview_frames[:4]

    # 간단 통계
    torso_mean, torso_std = float(torso_arr.mean()), float(torso_arr.std())
    knee_std = float(knee_arr.std()) if len(knee_arr) else 0.0
    elbow_std = float(elbow_arr.std()) if len(elbow_arr) else 0.0
    hip_std = float(hip_arr.std()) if len(hip_arr) else 0.0

    # 시계열 smoothing
    torso_sm = moving_average(torso_arr, window=5)
    knee_sm = moving_average(knee_arr, window=5)
    elbow_sm = moving_average(elbow_arr, window=5)

    exercise_key = "mixed"
    reps = 0
    metric_type = "reps"
    metric_value = 0.0
    posture_quality = "fair"
    intensity = "moderate"
    stability = "medium"
    notes = []

    # ----------- 분류 규칙 -----------

    # 1) 플랭크: 수평에 가깝고, 움직임이 거의 없음
    if torso_mean > 50 and torso_std < 8 and knee_std < 8 and elbow_std < 8 and hip_std < 0.01:
        exercise_key = "plank"
        metric_type = "seconds"
        metric_value = duration
        reps = 1
        posture_quality = "good"
        stability = "high"
        intensity = "moderate"
        notes.append("몸통과 엉덩이 움직임이 거의 없어 플랭크로 추정됩니다.")

    # 2) 푸시업: 수평 + 팔꿈치 각도 변화 큼 + 엉덩이 높이 변화는 상대적으로 적음
    elif torso_mean > 40 and elbow_std > 15 and hip_std < 0.05:
        exercise_key = "pushup"
        low, high = 100, 150
        reps = count_reps_from_series(elbow_sm, low_th=low, high_th=high)
        metric_type = "reps"
        metric_value = float(reps)
        posture_quality = "good" if hip_std < 0.03 else "fair"
        stability = "medium"
        intensity = "high" if reps >= 20 else "moderate"
        notes.append("수평 자세에서 팔꿈치 굽힘/펴짐 패턴이 반복되어 팔굽혀펴기로 추정됩니다.")

    # 3) 스쿼트: 수직 + 무릎 각도 변화 큼 + 엉덩이 높이 변화 큼
    elif torso_mean < 40 and knee_std > 15 and hip_std > 0.03:
        exercise_key = "squat"
        low, high = 140, 170  # 내려갈 때 각도 작아지고(굽힘), 올라올 때 커짐
        reps = count_reps_from_series(knee_sm * -1.0, low_th=-170, high_th=-140)  # 간단한 변형
        if reps == 0:
            # 그냥 knee_series로 직접 세기
            reps = count_reps_from_series(knee_sm, low_th=120, high_th=160)
        metric_type = "reps"
        metric_value = float(reps)
        posture_quality = "good" if knee_std > 25 else "fair"
        stability = "medium"
        intensity = "high" if reps >= 20 else "moderate"
        notes.append("무릎 굽힘/펴짐과 엉덩이 상하 이동이 반복되어 스쿼트로 추정됩니다.")

    # 4) 윗몸일으키기: 수평+수직 사이에서 몸통 각도 변화 큼, 무릎/엉덩이는 비교적 안정
    elif torso_std > 20 and knee_std < 10:
        exercise_key = "situp"
        low, high = 20, 60  # 누운 상태(수평에 가까움) ↔ 일어난 상태(수직에 가까움)
        reps = count_reps_from_series(torso_sm, low_th=low, high_th=high)
        metric_type = "reps"
        metric_value = float(reps)
        posture_quality = "fair"
        stability = "medium"
        intensity = "high" if reps >= 30 else "moderate"
        notes.append("상체 기울기 변화가 반복되어 윗몸일으키기로 추정됩니다.")

    # 5) 위 규칙에 안 걸리면 혼합 동작 처리
    else:
        exercise_key = "mixed"
        metric_type = "reps"
        # 가장 크게 움직인 각도로 임의 반복수 추정
        stds = [("torso", torso_std), ("knee", knee_std), ("elbow", elbow_std)]
        main_sig = max(stds, key=lambda x: x[1])[0]
        if main_sig == "torso":
            reps = count_reps_from_series(torso_sm, low_th=np.percentile(torso_sm, 30), high_th=np.percentile(torso_sm, 70))
        elif main_sig == "knee":
            reps = count_reps_from_series(knee_sm, low_th=np.percentile(knee_sm, 30), high_th=np.percentile(knee_sm, 70))
        else:
            reps = count_reps_from_series(elbow_sm, low_th=np.percentile(elbow_sm, 30), high_th=np.percentile(elbow_sm, 70))
        metric_value = float(max(0, reps))
        posture_quality = "unknown"
        stability = "medium"
        intensity = "moderate"
        notes.append("어떤 한 가지 운동 패턴으로 보기 어려워 혼합/기타 동작으로 분류했습니다.")

    analysis = {
        "exercise_key": exercise_key,
        "exercise_name_kr": EXERCISE_KEY_TO_NAME_KR.get(exercise_key, "알 수 없음"),
        "reps": int(reps),
        "metric_type": metric_type,
        "metric_value": float(metric_value),
        "posture_quality": posture_quality,
        "intensity": intensity,
        "stability": stability,
        "notes": " / ".join(notes),
        "video_duration_sec": float(duration),
        "stats": {
            "torso_mean": torso_mean,
            "torso_std": torso_std,
            "knee_std": knee_std,
            "elbow_std": elbow_std,
            "hip_std": hip_std,
        },
    }

    return analysis, preview_frames[:4]

# ============================================================
# 6. 점수 계산
# ============================================================

def lookup_kfta_score(exercise_key: str, gender: str, age_group: str, value: float) -> Tuple[int, int, str, str]:
    gender_key = "male" if gender == "남성" else "female"

    if exercise_key in NON_KFTA_EXERCISES or exercise_key not in KFTA_SCORES:
        max_ref = 50.0
        score = int(max(0, min(100, value / max_ref * 100)))
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
        remark = "해당 운동은 국민체력100 공식 항목이 아니거나 점수표가 없어 연구용 점수로 환산했습니다."
        return score, grade, level, remark

    table_exc = KFTA_SCORES.get(exercise_key, {})
    table_gender = table_exc.get(gender_key, {})
    thresholds = table_gender.get(age_group, [])

    if not thresholds:
        return 0, 0, "점수표 없음", "해당 연령/성별에 대한 점수표가 등록되어 있지 않습니다."

    score = 0
    for v_min, s in thresholds:
        if value >= v_min:
            score = s
            break

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

    remark = "점수표 수치는 예시값입니다. 실제 국민체력100 공식 기준으로 교체해 사용해야 합니다."
    return score, grade, level, remark

# ============================================================
# 7. Streamlit UI
# ============================================================

with st.sidebar:
    st.header("⚙️ 설정")
    age_group = st.selectbox("연령대", ["10대", "20대", "30대", "40대", "50대", "60대 이상"], index=1)
    gender = st.selectbox("성별", ["남성", "여성"], index=0)
    st.markdown("---")
    st.markdown(
        """
**분석 방식**

- MediaPipe Pose로 관절 추출
- 간단한 규칙 기반으로 운동 분류 및 반복 수 추정
- KFTA 예시 점수표로 점수 환산
"""
    )

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("1️⃣ 영상 업로드")
    video_file = st.file_uploader("운동 영상(mp4)", type=["mp4"])
    analyze_button = st.button("🔍 분석 실행", type="primary")

with col_right:
    st.subheader("2️⃣ 영상 미리보기")
    if video_file is not None:
        st.video(video_file)
    else:
        st.info("왼쪽에서 mp4 파일을 업로드하면 이곳에서 미리 볼 수 있습니다.")

st.markdown("---")

if analyze_button:
    if video_file is None:
        st.error("먼저 mp4 영상을 업로드해 주세요.")
        st.stop()

    video_bytes = video_file.getvalue()

    try:
        with st.spinner("🎞 MediaPipe Pose로 영상 분석 중..."):
            analysis, preview_frames = analyze_video_with_mediapipe(video_bytes)
    except Exception as e:
        st.error(f"영상 분석 중 오류가 발생했습니다: {e}")
        st.stop()

    st.success("분석 완료!")

    st.subheader("3️⃣ 대표 프레임")
    cols = st.columns(len(preview_frames))
    for i, frame in enumerate(preview_frames):
        cols[i].image(frame, caption=f"Frame {i+1}", use_container_width=True)

    st.markdown("---")
    st.subheader("4️⃣ 운동 분류 및 반복 수 추정")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("운동 분류", f"{analysis['exercise_name_kr']} ({analysis['exercise_key']})")
    with c2:
        if analysis["metric_type"] == "seconds":
            st.metric("유지 시간", f"{analysis['metric_value']:.1f} 초")
        else:
            st.metric("반복 수(추정)", f"{analysis['metric_value']:.0f} 회")
    with c3:
        st.metric("영상 길이", f"{analysis['video_duration_sec']:.1f} 초")

    st.write(
        f"- 자세 품질: **{analysis['posture_quality']}**  \n"
        f"- 강도: **{analysis['intensity']}**  \n"
        f"- 안정성: **{analysis['stability']}**"
    )

    if analysis["notes"]:
        st.info("추정 근거 / 코멘트: " + analysis["notes"])

    st.markdown("---")
    st.subheader("5️⃣ 국민체력100 점수 (예시 환산)")

    metric_value = analysis["metric_value"]
    # 플랭크는 초 단위 그대로, 나머지는 반복 수로 사용
    score, grade, level_label, remark = lookup_kfta_score(
        exercise_key=analysis["exercise_key"],
        gender=gender,
        age_group=age_group,
        value=metric_value,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("점수 (0~100)", f"{score} 점")
    with c2:
        if grade > 0:
            st.metric("등급", f"{grade} 등급")
        else:
            st.metric("등급", "기준 없음")
    with c3:
        st.metric("평가", level_label)

    st.caption(remark)

    st.markdown("---")
    st.subheader("6️⃣ 내부 분석 값 (디버그용)")
    st.json(analysis["stats"])
