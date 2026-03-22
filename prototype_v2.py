import streamlit as st
import pandas as pd
import random
import os
import cv2
from deepface import DeepFace
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCI Learning Lab v2",
    page_icon="🎓",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
STUDY_LOGS_FILE  = "study_logs.csv"
SAMPLE_DATA_FILE = "sample_data.csv"
CSV_COLUMNS      = ["timestamp", "student_id", "event_type",
                    "engagement_score", "emotion", "quiz_score"]

EMOTIONS_HIGH   = ["Engagement", "Delight"]
EMOTIONS_MID    = ["Surprise", "Confusion"]
EMOTIONS_LOW    = ["Boredom", "Frustration"]
ALL_EMOTIONS    = EMOTIONS_HIGH + EMOTIONS_MID + EMOTIONS_LOW

EMOTION_COLOR = {
    "Engagement":  "#4ade80",
    "Delight":     "#a78bfa",
    "Surprise":    "#60a5fa",
    "Confusion":   "#f59e0b",
    "Boredom":     "#94a3b8",
    "Frustration": "#f87171",
}

# Quiz: (question_text, options_dict, correct_key)
# ⚠️  correct_key is intentionally NOT shown in the UI — scoring is server-side
QUIZ = [
    (
        "Q1：「DK is being a big baby.」這句話表達的是？",
        {"A": "永久性的性格描述", "B": "此刻短暫的行為表現"},
        "B",
    ),
    (
        "Q2：強調「從過去持續到現在，且動作還在進行」應使用？",
        {"A": "現在完成進行式", "B": "現在完成式"},
        "A",
    ),
]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #0d0f14; color: #e8e6e0; }

.header-bar {
    background: linear-gradient(90deg, #1a1d26 0%, #12151e 100%);
    border-bottom: 1px solid #2a2d3a;
    padding: 16px 24px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 28px;
    border-radius: 0 0 12px 12px;
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem; color: #7eb8f7;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.header-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #4ade80; box-shadow: 0 0 8px #4ade80;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: #7eb8f7; margin-bottom: 8px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content:''; flex:1; height:1px;
    background: linear-gradient(90deg, #2a2d3a, transparent);
}

/* Live metrics bar */
.metrics-bar {
    background: #13161f; border: 1px solid #22263a; border-radius: 10px;
    padding: 10px 16px; margin-top: 10px;
    font-family: 'Space Mono', monospace; font-size: 0.75rem;
    display: flex; align-items: center; gap: 10px;
    flex-wrap: wrap;
}
.metrics-label { color: #4a4d5a; }
.metrics-score { color: #e8e6e0; font-weight: 700; }
.metrics-emotion {
    padding: 2px 10px; border-radius: 12px;
    font-size: 0.7rem; font-weight: 700;
    background: #1a1d26; border: 1px solid;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80; box-shadow: 0 0 6px #4ade80;
    animation: pulse 1.5s infinite; display: inline-block;
}

.log-row {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 12px; border-radius: 6px;
    background: #0d0f14; margin-bottom: 6px;
    font-size: 0.82rem; border-left: 3px solid transparent;
}
.log-row.start  { border-left-color: #4ade80; }
.log-row.pause  { border-left-color: #f59e0b; }
.log-row.submit { border-left-color: #7eb8f7; }
.log-event { font-family:'Space Mono',monospace; font-size:0.75rem; min-width:70px; }
.log-ts    { color:#4a4d5a; font-size:0.75rem; }
.log-eng   { color:#a78bfa; font-size:0.72rem; }

.status-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1a1d26; border: 1px solid #22263a; border-radius: 20px;
    padding: 4px 12px; font-family:'Space Mono',monospace;
    font-size: 0.7rem; color: #7eb8f7; margin-bottom: 20px;
}

/* ── Buttons: dark text on light background ─────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    width: 100%; border-radius: 8px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    padding: 10px 0; transition: all 0.2s; border: 1px solid;
    color: #000000 !important;       /* 將字體改為純黑色 */
    background: #e2e8f0 !important;  /* 將底色改為淺灰色 */
    border-color: #cbd5e1 !important;/* 邊框使用稍微深一點的灰色增加立體感 */
}
.stButton > button:hover:not(:disabled), .stDownloadButton > button:hover:not(:disabled) {
    background: #f1f5f9 !important;  /* 滑鼠懸停時變成更亮的淺灰 */
    color: #000000 !important;
}
.stButton > button:disabled, .stDownloadButton > button:disabled {
    opacity: 0.38;
    cursor: not-allowed;
}

.quiz-notice {
    background: #1a1d26; border: 1px solid #22263a; border-radius: 10px;
    padding: 14px 18px; text-align: center; margin-top: 12px;
    font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #7eb8f7;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_metrics() -> dict:
    """Generate correlated emotion + engagement score (fallback / placeholder)."""
    tier = random.choice(["high", "mid", "low"])
    if tier == "high":
        emotion = random.choice(EMOTIONS_HIGH)
        score   = round(random.uniform(0.75, 1.00), 2)
    elif tier == "mid":
        emotion = random.choice(EMOTIONS_MID)
        score   = round(random.uniform(0.40, 0.74), 2)
    else:
        emotion = random.choice(EMOTIONS_LOW)
        score   = round(random.uniform(0.00, 0.39), 2)
    return {"emotion": emotion, "engagement_score": score}


def calc_quiz_score(answers: dict) -> int:
    """Score quiz answers server-side: 50 pts per correct answer.
    Correct answers are defined only in QUIZ above — never exposed in the UI."""
    score = 0
    for i, (_, _, correct) in enumerate(QUIZ):
        key = f"q{i+1}"
        if answers.get(key) == correct:
            score += 50
    return score


def append_to_study_logs(row: dict):
    """Append one row to study_logs.csv (create with header if missing)."""
    df_row = pd.DataFrame([{c: row.get(c, "") for c in CSV_COLUMNS}])
    if os.path.exists(STUDY_LOGS_FILE):
        df_row.to_csv(STUDY_LOGS_FILE, mode="a", header=False, index=False)
    else:
        df_row.to_csv(STUDY_LOGS_FILE, mode="w", header=True, index=False)


def ensure_sample_data():
    """Auto-generate sample_data.csv with 5 rows if it doesn't exist."""
    if os.path.exists(SAMPLE_DATA_FILE):
        return
    sample_ids = ["S10001", "S10002", "S10003", "S10004", "S10005"]
    events     = ["START", "PAUSE", "START", "PAUSE", "SUBMIT"]
    rows = []
    for sid, ev in zip(sample_ids, events):
        m = generate_metrics()
        rows.append({
            "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id":       sid,
            "event_type":       ev,
            "engagement_score": m["engagement_score"],
            "emotion":          m["emotion"],
            "quiz_score":       calc_quiz_score({"q1": random.choice(["A", "B"]),
                                                 "q2": random.choice(["A", "B"])})
                                if ev == "SUBMIT" else "null",
        })
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(SAMPLE_DATA_FILE, index=False)


# ── Session state init ────────────────────────────────────────────────────────
if "logs"      not in st.session_state: st.session_state.logs      = []
if "status"    not in st.session_state: st.session_state.status    = "idle"
if "metrics"   not in st.session_state: st.session_state.metrics   = generate_metrics()
if "submitted" not in st.session_state: st.session_state.submitted = False

ensure_sample_data()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-bar">
    <div class="header-dot"></div>
    <span class="header-title">HCI Learning Lab &nbsp;/&nbsp; Prototype v2</span>
</div>
""", unsafe_allow_html=True)

# ── Student ID ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">🎓 Student Identity</div>', unsafe_allow_html=True)
student_id = st.text_input(
    label="Student ID",
    placeholder="請輸入學號 (e.g. B11234567，需先輸入學號才可解鎖影片播放與學習紀錄功能。) ",
    label_visibility="collapsed",
)
has_id = bool(student_id and student_id.strip())

if has_id:
    st.markdown(
        f'<div class="status-badge"><span style="color:#4ade80">●</span>'
        f' Logged in as &nbsp;<strong>{student_id.strip()}</strong></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="status-badge"><span style="color:#f59e0b">●</span>'
        ' Awaiting student ID … (please enter ID to enable controls)</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

# ── LEFT: Video + Controls ────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">📺 Learning Video</div>', unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=PIZSp-yNShk")

    st.markdown('<div class="section-label" style="margin-top:20px">🎮 Playback Controls</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    def capture_and_log(event_type: str, quiz_answers: dict):
        """Capture current metrics, compute quiz_score server-side, write row."""
        m  = st.session_state.metrics
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Score computed entirely in Python — result only written to CSV
        qs = calc_quiz_score(quiz_answers) if event_type == "SUBMIT" else "null"
        row = {
            "timestamp":        ts,
            "student_id":       student_id.strip(),
            "event_type":       event_type,
            "engagement_score": m["engagement_score"],
            "emotion":          m["emotion"],
            "quiz_score":       qs,
        }
        st.session_state.logs.append(row)
        append_to_study_logs(row)

    q_answers = {
        f"q{i+1}": st.session_state.get(f"quiz_q{i+1}", None)
        for i in range(len(QUIZ))
    }

    with b1:
        if st.button("▶  Start", key="btn_start",
                     use_container_width=True, disabled=not has_id):
            capture_and_log("START", q_answers)
            st.session_state.status = "playing"
            st.toast("▶ Session started", icon="🟢")

    with b2:
        if st.button("⏸  Pause", key="btn_pause",
                     use_container_width=True, disabled=not has_id):
            capture_and_log("PAUSE", q_answers)
            st.session_state.status = "paused"
            st.toast("⏸ Session paused", icon="🟡")

    with b3:
        if st.button("✔  Submit", key="btn_submit",
                     use_container_width=True, disabled=not has_id):
            capture_and_log("SUBMIT", q_answers)
            st.session_state.status    = "submitted"
            st.session_state.submitted = True
            st.success(f"✅ Log appended to `{STUDY_LOGS_FILE}`")
            st.toast("Submitted!", icon="💾")

# ── RIGHT: Webcam + Live Metrics + Quiz ──────────────────────────────────────
with col_right:

    # ── LIVE Webcam & Metrics ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">📷 Live Emotion Monitor</div>', unsafe_allow_html=True)

    # 1. 在 session_state 中初始化攝影機，避免重複開關
    if "camera" not in st.session_state:
        st.session_state.camera = cv2.VideoCapture(0)

    # 2. 利用 fragment 每 1.5 秒自動執行一次擷取與分析
    @st.fragment(run_every=1.5)
    def live_metrics_panel():
        cap = st.session_state.camera
        ret, frame = cap.read()

        # 顯示靜態佔位圖，取代原本的 st.image
        st.markdown("""
        <div style="background:#1a1d26; border: 1px dashed #4ade80; border-radius: 10px; padding: 30px 20px; text-align: center; margin-bottom: 15px;">
            <div style="font-size: 2.2rem; margin-bottom: 8px;">🤖</div>
            <div style="color: #4ade80; font-family: 'Space Mono', monospace; font-size: 0.85rem;">
                ● 情緒偵測運行中
            </div>
        </div>
        """, unsafe_allow_html=True)

        if ret:
            # 在背景默默將 Frame 丟給 DeepFace 分析
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                raw_emotion = result[0]["dominant_emotion"]

                MAP = {
                    "happy": "Delight", "neutral": "Engagement", "surprise": "Surprise",
                    "fear": "Confusion", "sad": "Boredom", "angry": "Frustration", "disgust": "Frustration"
                }
                detected = MAP.get(raw_emotion, "Engagement")

                # 根據情緒狀態給予動態的專注度分數
                if detected in EMOTIONS_HIGH:
                    score = round(random.uniform(0.75, 1.00), 2)
                elif detected in EMOTIONS_MID:
                    score = round(random.uniform(0.40, 0.74), 2)
                else:
                    score = round(random.uniform(0.00, 0.39), 2)

                st.session_state.metrics = {"emotion": detected, "engagement_score": score}

            except Exception:
                # 若抓不到臉，維持上次數據
                pass
        else:
            st.caption("🔴 無法讀取攝影機畫面，請確認硬體連線或權限。")

        # ── 渲染即時指標 UI (Metrics Bar) ──
        m       = st.session_state.metrics
        emotion = m["emotion"]
        score   = m["engagement_score"]
        color   = EMOTION_COLOR.get(emotion, "#7eb8f7")
        pct     = int(score * 100)
        bar_color = "#4ade80" if score >= 0.75 else ("#f59e0b" if score >= 0.40 else "#f87171")
        
        st.markdown(f"""
        <div class="metrics-bar">
            <span class="live-dot"></span>
            <span class="metrics-label">CURRENT STATUS</span>
            <span class="metrics-score">{score:.2f}</span>
            <span class="metrics-emotion" style="color:{color}; border-color:{color}40;">{emotion}</span>
        </div>
        <div style="margin-top:6px; height:5px; background:#1a1d26; border-radius:4px; overflow:hidden;">
            <div style="width:{pct}%; height:100%; background:{bar_color};
                        border-radius:4px; transition:width .5s ease;"></div>
        </div>
        """, unsafe_allow_html=True)

    # 執行 fragment
    live_metrics_panel()

    # ── Quiz ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:22px">💬 Quick Quiz</div>', unsafe_allow_html=True)

    quiz_disabled = st.session_state.submitted  # lock after submission

    for i, (q_text, options, _correct) in enumerate(QUIZ):
        # _correct intentionally ignored — never rendered in UI
        st.markdown(
            f'<p style="color:#c8c6c0;font-size:.95rem;margin-bottom:6px">'
            f'<strong>{q_text}</strong></p>',
            unsafe_allow_html=True,
        )
        st.radio(
            label=q_text,
            options=list(options.keys()),
            format_func=lambda k, o=options: f"({k}) {o[k]}",
            key=f"quiz_q{i+1}",
            label_visibility="collapsed",
            horizontal=True,
            disabled=quiz_disabled,
        )

    # Score is NEVER shown live. Only a submission-status notice is displayed.
    if st.session_state.submitted:
        st.markdown("""
        <div class="quiz-notice">
            ✅ &nbsp;Quiz submitted — your answers have been recorded.<br>
            <span style="color:#4a4d5a;font-size:0.72rem;margin-top:4px;display:block;">
                Score is stored in study_logs.csv only.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="quiz-notice" style="color:#4a4d5a;">
            📝 &nbsp;Complete the quiz and press
            <strong style="color:#7eb8f7">Submit</strong> to record your answers.
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# EVENT LOG
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📋 Event Log (this session)</div>', unsafe_allow_html=True)

if not st.session_state.logs:
    st.caption("No events recorded yet. Press Start, Pause, or Submit to begin logging.")
else:
    color_map = {"START": "start", "PAUSE": "pause", "SUBMIT": "submit"}
    icon_map  = {"START": "▶", "PAUSE": "⏸", "SUBMIT": "✔"}
    rows_html = ""
    for entry in reversed(st.session_state.logs):
        ev   = entry["event_type"]
        cls  = color_map.get(ev, "")
        icon = icon_map.get(ev, "•")
        ec   = EMOTION_COLOR.get(entry["emotion"], "#7eb8f7")
        # quiz_score deliberately omitted from the visible event log
        rows_html += f"""
        <div class="log-row {cls}">
            <span class="log-event">{icon} {ev}</span>
            <span class="log-ts">{entry['timestamp']}</span>
            <span class="log-eng" style="color:{ec}">{entry['emotion']} · {entry['engagement_score']}</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    df_session = pd.DataFrame(st.session_state.logs, columns=CSV_COLUMNS)
    st.download_button(
        label="⬇  Download Session CSV",
        data=df_session.to_csv(index=False).encode("utf-8"),
        file_name="study_logs_session.csv",
        mime="text/csv",
    )

# ── study_logs.csv viewer ─────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-label">🗂 study_logs.csv (all records)</div>', unsafe_allow_html=True)
if os.path.exists(STUDY_LOGS_FILE):
    df_all = pd.read_csv(STUDY_LOGS_FILE)
    st.dataframe(df_all, use_container_width=True, height=220)
    st.download_button(
        label="⬇  Download Full study_logs.csv",
        data=df_all.to_csv(index=False).encode("utf-8"),
        file_name=STUDY_LOGS_FILE,
        mime="text/csv",
    )
else:
    st.caption("study_logs.csv will appear here after the first event is logged.")

# ── sample_data.csv viewer ────────────────────────────────────────────────────
with st.expander("📄 sample_data.csv (auto-generated reference)"):
    if os.path.exists(SAMPLE_DATA_FILE):
        st.dataframe(pd.read_csv(SAMPLE_DATA_FILE), use_container_width=True)
