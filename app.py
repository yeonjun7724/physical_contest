import streamlit as st
import base64
import requests
import os
import json

# ------------------------------------------------------------
# 1) OpenAI API (REST 방식)
# ------------------------------------------------------------
def call_openai(messages, model="gpt-4o-mini"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 없습니다.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1500,
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ------------------------------------------------------------
# 2) VLM 분석 함수
# ------------------------------------------------------------
def analyze_video_with_vlm(video_bytes, duration_sec):
    b64 = base64.b64encode(video_bytes).decode()
    video_url = f"data:video/mp4;base64,{b64}"

    system_prompt = """
    당신은 국민체력100(국민체력인증센터) 전문 평가관입니다.
    사용자가 업로드한 운동 영상을 아래 기준에 따라 분석하고,
    JSON 형식으로 반환하세요.

    ① 반복 속도(페이스)
    ② 동작 정확성
    ③ 신체 정렬(척추/골반/무릎 정렬)
    ④ 상·하체 협응
    ⑤ 안정성(흔들림)
    ⑥ 반복수(예측 가능 시)

    JSON 예시:
    {
        "speed": "적절 | 빠름 | 느림",
        "accuracy": "우수 | 보통 | 부족",
        "alignment": "정상 | 틀어짐 | 불안정",
        "coordination": "우수 | 양호 | 부족",
        "stability": "안정적 | 흔들림 있음",
        "repetition_est": 24,
        "notes": "허리가 약간 후만됨, 양손 움직임 불규칙"
    }
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "운동 영상을 분석해 주세요."},
                {"type": "input_video", "video_url": video_url, "mime_type": "video/mp4"},
            ],
        },
    ]

    result = call_openai(messages)
    return json.loads(result)


# ------------------------------------------------------------
# 3) 국민체력100 기반 점수화 알고리즘
# ------------------------------------------------------------
def score_kfta(analysis):
    score = 0

    map_score = {
        "우수": 20,
        "정상": 20,
        "적절": 20,
        "양호": 20,
        "안정적": 20,

        "보통": 12,
        "다소 부족": 10,
        "흔들림 있음": 10,
        "느림": 10,
        "빠름": 10,

        "부족": 6,
        "불안정": 6,
        "틀어짐": 6
    }

    for k, v in analysis.items():
        if isinstance(v, str):
            score += map_score.get(v, 0)

    total = min(score, 100)

    if total >= 90:
        grade = "A"
    elif total >= 75:
        grade = "B"
    elif total >= 60:
        grade = "C"
    else:
        grade = "D"

    return total, grade


# ------------------------------------------------------------
# 4) Streamlit UX/UI + 기능
# ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="국민체력 100 AI 분석기",
        layout="wide",
    )

    st.markdown("""
    <style>
        .big-title { font-size: 32px; font-weight: 800; }
        .sub { color:#666; font-size:15px; }
        .box { padding:18px; border-radius:12px; background:#f8f9fa; border:1px solid #e5e7eb; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='big-title'>🏋️ 국민체력 100 - AI 운동 분석기</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub'>영상 기반 자동 분석 · VLM(gpt-4o-mini)</div>", unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns([1, 2])

    # ---------------- 좌측 설명 ----------------
    with col1:
        st.markdown("<div class='box'>", unsafe_allow_html=True)
        st.markdown("### 📌 분석 항목")
        st.markdown("""
        - 반복 속도(페이스)
        - 동작 정확성
        - 신체 정렬(척추/골반/무릎)
        - 상·하체 협응도
        - 안정성(흔들림)
        - 반복수 추정
        """)
        st.markdown("### 📌 계산 기준")
        st.markdown("""
        **국민체력100 공식 등급 체계 기반**
        - 90점 이상: A  
        - 75점 이상: B  
        - 60점 이상: C  
        - 그 이하: D  
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- 우측 영상 업로드 ----------------
    with col2:
        st.markdown("### 🎥 운동 영상 업로드")
        video = st.file_uploader("MP4 파일만 업로드 가능합니다.", type=["mp4"])

        duration_sec = st.number_input("📏 영상 길이(초)", 1, 300, 10)

        if video is not None:
            st.video(video)

        if video and st.button("🚀 AI 분석 시작"):
            st.info("영상 분석 중입니다… 약 10~20초 소요됩니다.")

            video_bytes = video.read()

            with st.spinner("VLM이 영상을 분석 중…"):
                analysis = analyze_video_with_vlm(video_bytes, duration_sec)

            st.success("분석 완료!")

            st.subheader("📊 분석 결과(JSON)")
            st.json(analysis)

            total, grade = score_kfta(analysis)

            # ---------------- 점수 카드 ----------------
            st.markdown("### 🏅 국민체력100 점수")
            st.metric("총점", f"{total} / 100")
            st.metric("등급", grade)

            # ---------------- AI 리포트 ----------------
            report_prompt = f"""
            당신은 국민체력100 전문 평가관입니다.
            아래 JSON을 기반으로 운동 평가 리포트를 작성하세요.

            {json.dumps(analysis, ensure_ascii=False)}

            요구사항:
            - 국민체력100 결과지 말투
            - 개선점 5가지
            - 속도·정확성·정렬·안정성·협응에 대한 평가
            - 훈련 팁 포함
            """

            messages = [
                {"role": "system", "content": "당신은 국민체력100 공식 평가관입니다."},
                {"role": "user", "content": report_prompt},
            ]

            with st.spinner("AI 리포트 생성 중…"):
                report = call_openai(messages)

            st.subheader("📄 AI 코치 리포트")
            st.write(report)


# ------------------------------------------------------------
if __name__ == "__main__":
    main()
