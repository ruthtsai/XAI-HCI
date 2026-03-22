import streamlit as st
import pandas as pd
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCI Learning Lab",
    page_icon="🎓",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0d0f14;
    color: #e8e6e0;
}

/* Header strip */
.header-bar {
    background: linear-gradient(90deg, #1a1d26 0%, #12151e 100%);
    border-bottom: 1px solid #2a2d3a;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 28px;
    border-radius: 0 0 12px 12px;
}
.header-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #7eb8f7;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.header-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* Section labels */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #7eb8f7;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2a2d3a, transparent);
}

/* Cards */
.card {
    background: #13161f;
    border: 1px solid #22263a;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Webcam placeholder */
.webcam-placeholder {
    background: linear-gradient(135deg, #0d0f14 0%, #1a1d26 100%);
    border: 1px dashed #2a2d3a;
    border-radius: 10px;
    height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: #4a4d5a;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
}
.webcam-icon {
    font-size: 2.5rem;
    opacity: 0.4;
}
.webcam-badge {
    background: #1e2130;
    border: 1px solid #2a2d3a;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.65rem;
    color: #f59e0b;
    letter-spacing: 0.12em;
}

/* Log table */
.log-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border-radius: 6px;
    background: #0d0f14;
    margin-bottom: 6px;
    font-size: 0.82rem;
    border-left: 3px solid transparent;
}
.log-row.start  { border-left-color: #4ade80; }
.log-row.pause  { border-left-color: #f59e0b; }
.log-row.submit { border-left-color: #7eb8f7; }
.log-event {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    min-width: 70px;
}
.log-ts { color: #4a4d5a; font-size: 0.75rem; }

/* Question area */
.question-text {
    font-size: 1rem;
    color: #c8c6c0;
    line-height: 1.6;
    margin-bottom: 14px;
}

/* Buttons – override streamlit defaults */
div[data-testid="column"] .stButton > button {
    width: 100%;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 10px 0;
    transition: all 0.2s;
    border: 1px solid;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1a1d26;
    border: 1px solid #22263a;
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #7eb8f7;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "logs" not in st.session_state:
    st.session_state.logs = []
if "status" not in st.session_state:
    st.session_state.status = "idle"   # idle | playing | paused | submitted
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# ── Helper ────────────────────────────────────────────────────────────────────
def capture_and_log(event_type: str):
    """將紀錄存入 session_state (純前端暫存，不寫入實體檔案)"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "student_id": student_id.strip() if student_id else "",
        "event": event_type,
        "timestamp": ts
    }
    st.session_state.logs.append(row)
    return True

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-bar">
    <div class="header-dot"></div>
    <span class="header-title">HCI Learning Lab &nbsp;/&nbsp; Baseline Prototype</span>
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

if student_id:
    st.markdown(f"""
    <div class="status-badge">
        <span style="color:#4ade80">●</span> Logged in as &nbsp;<strong>{student_id}</strong>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-badge">
        <span style="color:#f59e0b">●</span> Awaiting student ID …
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT  (video | webcam + Q&A)
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

# ── LEFT: Video ───────────────────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">📺 Learning Video</div>', unsafe_allow_html=True)
    with st.container():
        st.video("https://www.youtube.com/watch?v=PIZSp-yNShk")

    # Control buttons
    st.markdown('<div class="section-label" style="margin-top:20px">🎮 Playback Controls</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("▶  Start", key="btn_start", use_container_width=True, disabled=not has_id):
            if capture_and_log("START"):
                st.session_state.status = "playing"
                st.toast("▶ 學習紀錄已開始", icon="🟢")

    with b2:
        if st.button("⏸  Pause", key="btn_pause", use_container_width=True, disabled=not has_id):
            if capture_and_log("PAUSE"):
                st.session_state.status = "paused"
                st.toast("⏸ 學習紀錄已暫停", icon="🟡")

    with b3:
        if st.button("✔  Submit", key="btn_submit", use_container_width=True, disabled=not has_id):
            if capture_and_log("SUBMIT"):
                st.session_state.status = "submitted"
                st.session_state.submitted = True
                st.toast("✔ 紀錄已成功提交！", icon="💾")

# ── RIGHT: Webcam + Q&A ───────────────────────────────────────────────────────
with col_right:

    # Webcam placeholder
    st.markdown('<div class="section-label">📷 Webcam Monitor</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="webcam-placeholder">
        <div class="webcam-icon">📷</div>
        <span>CAMERA FEED RESERVED</span>
        <div class="webcam-badge">PLACEHOLDER</div>
    </div>
    """, unsafe_allow_html=True)

    # Q&A section
    st.markdown('<div class="section-label" style="margin-top:20px">💬 Quick Quiz</div>', unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# EVENT LOG
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">📋 Event Log</div>', unsafe_allow_html=True)

if not st.session_state.logs:
    st.caption("No events recorded yet. Press Start, Pause, or Submit to begin logging.")
else:
    color_map = {"START": "start", "PAUSE": "pause", "SUBMIT": "submit"}
    icon_map  = {"START": "▶", "PAUSE": "⏸", "SUBMIT": "✔"}
    rows_html = ""
    for entry in reversed(st.session_state.logs):
        ev   = entry["event"]
        cls  = color_map.get(ev, "")
        icon = icon_map.get(ev, "•")
        rows_html += f"""
        <div class="log-row {cls}">
            <span class="log-event">{icon} {ev}</span>
            <span class="log-ts">{entry['timestamp']}</span>
        </div>"""
    st.markdown(rows_html, unsafe_allow_html=True)

    # Download button
    # ── CSV 資料預覽與下載區塊 ─────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">🗂 Session Logs Data</div>', unsafe_allow_html=True)
    
    df_preview = pd.DataFrame(st.session_state.logs)
    
    if not df_preview.empty:
        # 在畫面上顯示一個類似 CSV 的表格區塊
        st.dataframe(df_preview, use_container_width=True, height=200)
        
        # 將記憶體中的 DataFrame 轉換成 CSV 格式供下載
        csv_bytes = df_preview.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇  Download CSV",
            data=csv_bytes,
            file_name="baseline_prototype.csv",
            mime="text/csv",
        )
    else:
        st.caption("尚無紀錄，點擊上方的 Start / Pause / Submit 開始產生資料。")
