import streamlit as st
import pandas as pd
import random
import os
import json
import cv2
from deepface import DeepFace
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCI Learning Lab v3",
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

EMOTIONS_HIGH = ["Engagement", "Delight"]
EMOTIONS_MID  = ["Surprise", "Confusion"]
EMOTIONS_LOW  = ["Boredom", "Frustration"]
ALL_EMOTIONS  = EMOTIONS_HIGH + EMOTIONS_MID + EMOTIONS_LOW

EMOTION_COLOR = {
    "Engagement":  "#4ade80",
    "Delight":     "#a78bfa",
    "Surprise":    "#60a5fa",
    "Confusion":   "#f59e0b",
    "Boredom":     "#94a3b8",
    "Frustration": "#f87171",
}

# Quiz：題目、選項、正確答案 (正確答案僅用於伺服器端計分，不在 UI 顯示)
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

GROQ_MODEL = "openai/gpt-oss-120b"

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

/* AI 助手提示 */
.ai-hint-box {
    background: linear-gradient(135deg, #0f1a2e 0%, #1a1d26 100%);
    border: 1px solid #2a4a6a;
    border-left: 3px solid #7eb8f7;
    border-radius: 10px;
    padding: 14px 16px;
    margin-top: 14px;
}
.ai-hint-title {
    font-family: 'Space Mono', monospace; font-size: 0.68rem;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: #7eb8f7; margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
}
.ai-hint-content {
    font-size: 0.88rem; color: #c8d8e8; line-height: 1.6;
}

/* Trace 區域 */
.trace-entry {
    font-family: 'Space Mono', monospace; font-size: 0.74rem;
    color: #6b8fa8; margin-bottom: 8px; padding: 6px 10px;
    background: #0a0d12; border-radius: 6px;
    border-left: 2px solid #2a4a6a;
    white-space: pre-wrap;
}
.trace-tool {
    color: #f59e0b; font-weight: 700;
}
.trace-reason {
    color: #a78bfa;
}

/* 底部按鈕區 */
.bottom-controls {
    position: sticky;
    bottom: 0;
    background: #0d0f14;
    border-top: 1px solid #22263a;
    padding: 14px 0 8px 0;
    z-index: 100;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def generate_metrics() -> dict:
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
    score = 0
    for i, (_, _, correct) in enumerate(QUIZ):
        key = f"q{i+1}"
        if answers.get(key) == correct:
            score += 50
    return score


def safe_append_to_study_logs(row: dict) -> bool:
    """回傳 True=成功, False=失敗 (PermissionError 等)"""
    try:
        df_row = pd.DataFrame([{c: row.get(c, "") for c in CSV_COLUMNS}])
        if os.path.exists(STUDY_LOGS_FILE):
            df_row.to_csv(STUDY_LOGS_FILE, mode="a", header=False, index=False)
        else:
            df_row.to_csv(STUDY_LOGS_FILE, mode="w", header=True, index=False)
        return True
    except PermissionError:
        st.error("⚠️ 儲存失敗：請關閉 CSV 檔案後重試（可能被 Excel 開啟中）")
        return False
    except Exception as e:
        st.error(f"⚠️ 儲存失敗：{e}")
        return False


def ensure_sample_data():
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
            "quiz_score":       calc_quiz_score({"q1": random.choice(["A","B"]),
                                                 "q2": random.choice(["A","B"])})
                                if ev == "SUBMIT" else "null",
        })
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(SAMPLE_DATA_FILE, index=False)


# ══════════════════════════════════════════════════════════════════════════════
# AGENT  ── 工具定義 + 推理迴圈
# ══════════════════════════════════════════════════════════════════════════════

def tool_read_logs() -> dict:
    """工具 1：讀取 study_logs.csv，回傳結構化摘要"""
    trace_log("TOOL_CALL", "read_logs", "呼叫工具：讀取 study_logs.csv")
    if not os.path.exists(STUDY_LOGS_FILE):
        result = {"status": "no_data", "message": "study_logs.csv 尚未建立"}
        trace_log("TOOL_RESULT", "read_logs", f"結果：{result}")
        return result

    df = pd.read_csv(STUDY_LOGS_FILE)
    if df.empty:
        result = {"status": "empty", "message": "CSV 存在但無任何紀錄"}
        trace_log("TOOL_RESULT", "read_logs", f"結果：{result}")
        return result

    latest  = df.iloc[-1]
    submits = df[df["event_type"] == "SUBMIT"]
    last_score = int(submits.iloc[-1]["quiz_score"]) if not submits.empty else None

    # 低參與度計算
    numeric_eng = pd.to_numeric(df["engagement_score"], errors="coerce")
    low_engagement_count = int((numeric_eng < 0.40).sum())

    # Q2 錯誤偵測（score < 100 代表至少一題錯）
    q2_wrong = last_score is not None and last_score < 100

    result = {
        "status":               "ok",
        "total_records":        len(df),
        "latest_emotion":       str(latest.get("emotion", "unknown")),
        "latest_engagement":    float(latest.get("engagement_score", 0)),
        "last_quiz_score":      last_score,
        "q2_likely_wrong":      q2_wrong,
        "low_engagement_count": low_engagement_count,
    }
    trace_log("TOOL_RESULT", "read_logs", json.dumps(result, ensure_ascii=False, indent=2))
    return result


def tool_recommend_hint(log_data: dict) -> str:
    """工具 2：根據 log 資料，決定要給哪個 hint (規則 + API)"""
    trace_log("TOOL_CALL", "recommend_hint", f"呼叫工具：根據 log 資料推薦提示 → {log_data}")

    # ── 規則引擎：先判斷是否需要 API ──
    low_eng     = log_data.get("latest_engagement", 1.0) < 0.40
    q2_wrong    = log_data.get("q2_likely_wrong", False)
    no_data     = log_data.get("status") != "ok"
    score       = log_data.get("last_quiz_score")

    if no_data:
        hint = "目前還沒有學習紀錄。請先點擊「Start」開始課程，記錄你的學習狀態。"
        trace_log("DECISION", "recommend_hint", "尚無資料 → 給予引導提示（不呼叫 API）")
        return hint

    # ── 呼叫 Groq API 產生個性化提示 ──
    context_parts = []
    if q2_wrong:
        context_parts.append("學生對 Q2（現在完成進行式）的回答可能有誤（總分未達 100 分）")
    if low_eng:
        context_parts.append(f"偵測到低參與度（分數：{log_data['latest_engagement']:.2f}，情緒：{log_data['latest_emotion']}）")
    if score is not None:
        context_parts.append(f"最近一次測驗分數：{score}/100")
    if not context_parts:
        context_parts.append(f"學生目前狀態良好（情緒：{log_data['latest_emotion']}，專注度：{log_data['latest_engagement']:.2f}）")

    situation = "；".join(context_parts)
    trace_log("DECISION", "recommend_hint",
              f"情境判斷完成 → {'低參與/答錯，呼叫 Groq API 生成個性化提示' if (q2_wrong or low_eng) else '學習狀態正常，呼叫 Groq API 給予鼓勵提示'}")

    try:
        from groq import Groq
        client = Groq()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一位溫暖、耐心的英語學習教學助理。"
                        "你的目標是提供適應性引導以支持學習，而非取代老師。"
                        "請根據學生的學習狀態，給予簡短（2-3 句）、具體的學習提示。"
                        "使用繁體中文回覆。語氣要親切，避免說教感。"
                        "不要在開頭稱呼學生，直接給提示內容。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"學生目前的學習狀況：{situation}。\n"
                        "請給予一個針對這個狀況的學習提示或鼓勵，"
                        "幫助學生理解課程內容（DK 的行為表現描述與現在完成進行式）。"
                    ),
                },
            ],
        )
        hint = response.choices[0].message.content.strip()
        trace_log("API_RESPONSE", "Groq", f"API 回應成功，tokens used: {response.usage.completion_tokens}")
    except Exception as e:
        # Fallback：規則型提示
        trace_log("API_ERROR", "Groq", f"API 呼叫失敗：{e}，切換至規則型提示")
        if q2_wrong and low_eng:
            hint = ("偵測到你在 Q2 遇到困難，且目前的專注度稍低。"
                    "「現在完成進行式」強調從過去開始、持續到現在的動作，"
                    "試著對比：『I have studied』(已完成) vs『I have been studying』(還在進行)。")
        elif q2_wrong:
            hint = ("Q2 的關鍵在於：現在完成進行式 = have/has been + V-ing，"
                    "強調動作『仍在持續』。例如：『She has been crying for an hour.』")
        elif low_eng:
            hint = (f"注意到你目前的狀態是 {log_data['latest_emotion']}，稍微休息一下也沒關係！"
                    "準備好後，我們再來看 DK 的例句，一起找出 be 動詞的規律。")
        else:
            hint = ("你的學習狀態不錯！記住：be 動詞加 ing 描述的是當下的行為，"
                    "而非永久的性格。試試看用英文描述你現在正在做的事。")

    return hint


# ── Trace 系統 ────────────────────────────────────────────────────────────────
def trace_log(trace_type: str, tool_name: str, message: str):
    """將 trace 紀錄加入 session_state"""
    entry = {
        "time":  datetime.now().strftime("%H:%M:%S"),
        "type":  trace_type,
        "tool":  tool_name,
        "msg":   message,
    }
    st.session_state.agent_trace.append(entry)


def run_agent_reasoning_loop() -> str:
    """
    Agent 推理迴圈（模擬 while 迴圈邏輯）：
    Step 1 → 查看 logs
    Step 2 → 判斷是否需要給予提示
    Step 3 → 呼叫 recommend_hint 工具
    """
    trace_log("AGENT_START", "教學助理 Agent", "開始推理迴圈")

    # Step 1: 先查看 logs
    trace_log("REASONING", "Agent", "Step 1：先查看學習紀錄，了解學生現況")
    log_data = tool_read_logs()

    # Step 2: 判斷
    needs_hint = True  # 教學助理永遠嘗試給予有意義的回饋
    if log_data.get("status") == "ok":
        low_eng  = log_data.get("latest_engagement", 1.0) < 0.40
        q2_wrong = log_data.get("q2_likely_wrong", False)
        if low_eng and q2_wrong:
            reason = "偵測到 Q2 答錯且參與度低，決定推薦分段提示"
        elif q2_wrong:
            reason = "偵測到 Q2 可能答錯，推薦語法解析提示"
        elif low_eng:
            reason = f"偵測到低參與度（{log_data.get('latest_emotion')}），推薦重燃動機提示"
        else:
            reason = "學生狀態良好，推薦正向強化提示"
    else:
        reason = "尚無學習紀錄，提供課程引導提示"

    trace_log("REASONING", "Agent", f"Step 2：{reason}")

    # Step 3: 生成提示
    hint = tool_recommend_hint(log_data)
    trace_log("AGENT_END", "教學助理 Agent", f"推理迴圈結束，已生成提示（{len(hint)} 字）")
    return hint


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "logs"        not in st.session_state: st.session_state.logs        = []
if "status"      not in st.session_state: st.session_state.status      = "idle"
if "metrics"     not in st.session_state: st.session_state.metrics     = generate_metrics()
if "submitted"   not in st.session_state: st.session_state.submitted   = False
if "agent_hint"  not in st.session_state: st.session_state.agent_hint  = None
if "agent_trace" not in st.session_state: st.session_state.agent_trace = []
if "camera"      not in st.session_state: st.session_state.camera      = cv2.VideoCapture(0)

ensure_sample_data()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-bar">
    <div class="header-dot"></div>
    <span class="header-title">HCI Learning Lab &nbsp;/&nbsp; Prototype v3 — Agent Edition</span>
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
# MAIN LAYOUT  ── 2 : 1 分欄
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([2, 1], gap="large")

# ── LEFT (2)：Learning Video ──────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="section-label">📺 Learning Video</div>', unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=PIZSp-yNShk")

# ── RIGHT (1)：Webcam → Metrics → Quiz → AI Hint ──────────────────────────────
with col_right:

    # ── 1. Webcam 佔位圖（頂端獨立，不重疊左側）──────────────────────────────
    st.markdown('<div class="section-label">📷 Live Emotion Monitor</div>', unsafe_allow_html=True)

    @st.fragment(run_every=1.5)
    def live_metrics_panel():
        cap = st.session_state.camera
        ret, frame = cap.read()

        st.markdown("""
        <div style="background:#1a1d26; border: 1px dashed #4ade80; border-radius: 10px;
                    padding: 28px 20px; text-align: center; margin-bottom: 12px;">
            <div style="font-size: 2rem; margin-bottom: 6px;">🤖</div>
            <div style="color: #4ade80; font-family: 'Space Mono', monospace; font-size: 0.8rem;">
                ● 情緒偵測運行中
            </div>
        </div>
        """, unsafe_allow_html=True)

        if ret:
            try:
                result   = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                raw_emo  = result[0]["dominant_emotion"]
                MAP = {
                    "happy": "Delight", "neutral": "Engagement", "surprise": "Surprise",
                    "fear":  "Confusion", "sad": "Boredom", "angry": "Frustration", "disgust": "Frustration"
                }
                detected = MAP.get(raw_emo, "Engagement")
                if detected in EMOTIONS_HIGH:
                    score = round(random.uniform(0.75, 1.00), 2)
                elif detected in EMOTIONS_MID:
                    score = round(random.uniform(0.40, 0.74), 2)
                else:
                    score = round(random.uniform(0.00, 0.39), 2)
                st.session_state.metrics = {"emotion": detected, "engagement_score": score}
            except Exception:
                pass
        else:
            st.caption("🔴 無法讀取攝影機，請確認硬體連線。")

        # ── 2. 模擬數據標籤（Metrics Bar）──────────────────────────────────
        m         = st.session_state.metrics
        emotion   = m["emotion"]
        score     = m["engagement_score"]
        color     = EMOTION_COLOR.get(emotion, "#7eb8f7")
        pct       = int(score * 100)
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

    live_metrics_panel()

    # ── 3. Quiz Area ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:20px">💬 Quiz Area</div>', unsafe_allow_html=True)

    quiz_disabled = st.session_state.submitted

    for i, (q_text, options, _correct) in enumerate(QUIZ):
        st.markdown(
            f'<p style="color:#c8c6c0;font-size:.9rem;margin-bottom:4px">'
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

    if st.session_state.submitted:
        st.markdown("""
        <div class="quiz-notice">
            ✅ &nbsp;Quiz submitted — answers recorded.<br>
            <span style="color:#4a4d5a;font-size:0.7rem;display:block;margin-top:4px;">
                Score is stored in study_logs.csv only.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="quiz-notice" style="color:#4a4d5a;">
            📝 &nbsp;Complete the quiz and press
            <strong style="color:#7eb8f7">Submit</strong> to record.
        </div>
        """, unsafe_allow_html=True)

    # ── 4. AI 學習助手提示（Quiz 下方）──────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:20px">🤖 AI Learning Assistant</div>', unsafe_allow_html=True)

    col_ai_btn, _ = st.columns([2, 1])
    with col_ai_btn:
        if st.button("🔍 取得 AI 提示", key="btn_agent", use_container_width=True):
            # 清空上次 trace
            st.session_state.agent_trace = []
            with st.spinner("Agent 推理中…"):
                hint = run_agent_reasoning_loop()
                st.session_state.agent_hint = hint

    if st.session_state.agent_hint:
        st.markdown(f"""
        <div class="ai-hint-box">
            <div class="ai-hint-title">✦ AI 學習助手提示</div>
            <div class="ai-hint-content">{st.session_state.agent_hint}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ai-hint-box" style="opacity:0.5;">
            <div class="ai-hint-title">✦ AI 學習助手提示</div>
            <div class="ai-hint-content" style="color:#4a4d5a;">
                點擊「取得 AI 提示」，讓助手分析你的學習狀態並給予建議。
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 底部操作區 ── 3 個核心按鈕，橫向排列
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown('<div class="section-label">🎮 Playback Controls</div>', unsafe_allow_html=True)

b1, b2, b3 = st.columns(3)

def capture_and_log(event_type: str, quiz_answers: dict):
    m  = st.session_state.metrics
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    return safe_append_to_study_logs(row)

q_answers = {
    f"q{i+1}": st.session_state.get(f"quiz_q{i+1}", None)
    for i in range(len(QUIZ))
}

with b1:
    if st.button("▶  Start", key="btn_start",
                 use_container_width=True, disabled=not has_id):
        if capture_and_log("START", q_answers):
            st.session_state.status = "playing"
            st.toast("▶ Session started", icon="🟢")

with b2:
    if st.button("⏸  Pause", key="btn_pause",
                 use_container_width=True, disabled=not has_id):
        if capture_and_log("PAUSE", q_answers):
            st.session_state.status = "paused"
            st.toast("⏸ Session paused", icon="🟡")

with b3:
    if st.button("✔  Submit", key="btn_submit",
                 use_container_width=True, disabled=not has_id):
        ok = capture_and_log("SUBMIT", q_answers)
        if ok:
            st.session_state.status    = "submitted"
            st.session_state.submitted = True
            st.toast("✔ Submitted successfully!", icon="💾")

# ══════════════════════════════════════════════════════════════════════════════
# AGENT TRACE  ── 可解釋性紀錄
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
with st.expander("🔬 Agent Trace（可解釋性紀錄）", expanded=False):
    if not st.session_state.agent_trace:
        st.caption("尚無 Agent 推理紀錄。點擊「取得 AI 提示」後，推理過程將顯示於此。")
    else:
        TYPE_COLOR = {
            "AGENT_START": "#4ade80",
            "AGENT_END":   "#4ade80",
            "TOOL_CALL":   "#f59e0b",
            "TOOL_RESULT": "#60a5fa",
            "REASONING":   "#a78bfa",
            "DECISION":    "#f97316",
            "API_RESPONSE":"#34d399",
            "API_ERROR":   "#f87171",
        }
        for entry in st.session_state.agent_trace:
            color = TYPE_COLOR.get(entry["type"], "#7eb8f7")
            st.markdown(f"""
            <div class="trace-entry">
                <span style="color:#4a4d5a">[{entry['time']}]</span>
                <span style="color:{color}; font-weight:700;">[{entry['type']}]</span>
                <span style="color:#94a3b8"> {entry['tool']} </span>→
                <span style="color:#c8d8e8"> {entry['msg']}</span>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# EVENT LOG
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
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
with st.expander("📄 sample_data.csv（自動產生的參考資料）"):
    if os.path.exists(SAMPLE_DATA_FILE):
        st.dataframe(pd.read_csv(SAMPLE_DATA_FILE), use_container_width=True)
