import streamlit as st
import pandas as pd
import uuid
import time
import os
import cv2
import numpy as np
from datetime import datetime
from deepface import DeepFace

# ── 頁面設定 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="教學實驗平台 (Version B：抽象解釋版)",
    page_icon="🎓",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 環境設定與初始化 (Env & Session)
# ══════════════════════════════════════════════════════════════════════════════
LOG_FILENAME = "study_logs_B.csv"

EMOTION_COLOR = {
    "Engagement":  "#4ade80",
    "Delight":     "#a78bfa",
    "Surprise":    "#60a5fa",
    "Confusion":   "#f59e0b",
    "Boredom":     "#94a3b8",
    "Frustration": "#f87171",
}

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_event_time" not in st.session_state:
    st.session_state.last_event_time = time.time()
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "agent_hint" not in st.session_state:
    st.session_state.agent_hint = ""
if "reasoning_trace" not in st.session_state:
    st.session_state.reasoning_trace = []
if "last_detected_metrics" not in st.session_state:
    st.session_state.last_detected_metrics = {"emotion": "Neutral", "score": 0}
if "camera_obj" not in st.session_state:
    st.session_state.camera_obj = cv2.VideoCapture(0)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .hint-box {
        background-color: #f0f7ff;
        border-left: 5px solid #007bff;
        padding: 15px;
        margin: 10px 0;
        border-radius: 4px;
        color: #004085;
    }
    .trace-card {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 10px;
        font-family: monospace;
        border-radius: 5px;
        font-size: 0.85rem;
    }
    .stButton > button {
        width: 100%;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between; /* 讓左右兩個項目推至兩端 */
        align-items: center;           /* 垂直居中對齊 */
        background-color: #13161f;     /* 深色背景 */
        padding: 15px 20px;
        border-radius: 10px;
        border: 1px solid #22263a;
        margin: 10px 0;
    }

    .metrics-item {
        display: flex;
        flex-direction: column;        /* 標籤與數值上下排列 */
    }

    .metrics-label {
        color: #e8e6e0;                /* 淺灰色文字 */
        font-size: 0.9rem;
        margin-bottom: 4px;
        font-weight: 500;
    }

    .metrics-value {
        font-size: 1.2rem;
        font-weight: bold;
        font-family: 'Space Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. 核心方法 (Decision Loop & Logging)
# ══════════════════════════════════════════════════════════════════════════════
def analyze_frame(frame):
    """分析單幀影像的情緒與參與度"""
    try:
        # DeepFace 分析
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        raw_emo = results[0]['dominant_emotion']
        
        # 映射情緒
        MAP = {
            "happy": "Delight", "neutral": "Engagement", "surprise": "Surprise",
            "fear": "Confusion", "sad": "Boredom", "angry": "Frustration"
        }
        detected = MAP.get(raw_emo, "Engagement")
        
        # 參與度計算：以情緒信賴度為基礎
        score = int(results[0]['emotion'][raw_emo])
        if raw_emo == "neutral": 
            # 平滑化 Neutral 的參與度顯示
            score = max(40, min(85, score))
            
        return detected, score
    except:
        return st.session_state.last_detected_metrics["emotion"], st.session_state.last_detected_metrics["score"]
    
def get_current_step():
    """動態判斷學生目前操作步驟"""
    if not st.session_state.quiz_started:
        return "Watching Video"
    q1 = st.session_state.get("q1_input", "")
    q2 = st.session_state.get("q2_input", "")
    if q1 == "":
        return "Q1"
    return "Q2"

def read_event_log():
    """計算 idle_time (秒)"""
    return round(time.time() - st.session_state.last_event_time, 2)

def write_log(event_type, hint_requested=False, score=None):
    """寫入符合 Schema 的日誌紀錄"""
    idle = read_event_log()
    current_step = get_current_step()
    m = st.session_state.last_detected_metrics
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_id": st.session_state.get("student_id", "Unknown"),
        "session_id": st.session_state.session_id,
        "environment": st.session_state.get("env_mode", "Classroom"),
        "event_type": event_type,
        "current_step": current_step,
        "emotion": m["emotion"],
        "engagement_score": m["score"],
        "idle_time": idle,
        "hint_requested": hint_requested,
        "hint_type": "Abstract Explanation",
        "quiz_score": score if score is not None else ""
    }
    st.session_state.logs.append(log_entry)
    st.session_state.last_event_time = time.time() # 更新最後活動時間
    
    # 持久化儲存
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILENAME, mode='a', header=not os.path.exists(LOG_FILENAME), index=False)

def generate_hint():
    """[核心邏輯] Rule-based Agent with Traceability"""
    idle = read_event_log()
    step = get_current_step()
    q1_val = st.session_state.get("q1_input", "")
    q2_val = st.session_state.get("q2_input", "")
    
    trace = {"Observation": f"idle_time={idle}s, step={step}, input='{q1_val if step=='Q1' else q2_val}'", "Analysis": "", "Action": ""}
    hint = ""

    # 條件 1 (卡住)
    if idle > 30 and step == "Q1" and q1_val == "":
        trace["Analysis"] = "學生可能未能區分「常態性格」與「短暫反常行為」的文法差異。"
        hint = "【概念提醒】請回想一下，當我們要形容一個人『此刻反常的短暫行為』時，通常會結合哪種動詞型態來表達狀態的持續？"
    
    # 條件 2 (答錯)
    elif step == "Q2" and q2_val != "" and "been" not in q2_val:
        trace["Analysis"] = "學生遺漏了現在完成進行式中強調「過程持續性」的元素。"
        hint = "【概念提醒】請檢視目前的答案。這句話想要強調動作『從過去一直持續到現在』的過程，請思考時態結構是否完整包含了『完成』與『進行』的元素。"
    
    # 條件 3 (預設)
    else:
        trace["Analysis"] = "normal behavior."
        hint = "【學習指引】學習新文法時，多留意句子想要強調的『時間狀態』與『動作持續性』。"

    trace["Action"] = hint
    st.session_state.reasoning_trace.append(trace)
    st.session_state.agent_hint = hint
    write_log("Hint", hint_requested=True)

def summarize_session():
    """結算測驗並儲存結果"""
    q1 = st.session_state.get("q1_input", "").lower()
    q2 = st.session_state.get("q2_input", "").lower()
    
    score = 0
    if "being" in q1: score += 50
    if "been" in q2 and ("teaching" in q2 or "taught" in q2): score += 50
    
    write_log("Submit", score=score)
    return score

# ══════════════════════════════════════════════════════════════════════════════
# 3. UI 佈局實作
# ══════════════════════════════════════════════════════════════════════════════

# 頂部導航與設定
st.title("🎓 教學實驗平台 (Version B：抽象解釋版)")
t1, t2 = st.columns([1, 1])
with t1:
    st.session_state.student_id = st.text_input("學號 Student ID", placeholder="B11234567")
with t2:
    st.session_state.env_mode = st.radio("實驗環境設定", ["Classroom", "Home"], horizontal=True)

st.divider()

# 主內容區：左側影片，右側 Webcam 與 測驗
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("📺 學習影片區域")
    st.video("https://www.youtube.com/watch?v=PIZSp-yNShk") # 範例文法影片

with col_right:
    st.subheader("📷 即時影像監測")
    @st.fragment(run_every=2)
    def live_metrics_panel():
        cap = st.session_state.camera_obj
        ret, frame = cap.read()
        
        if ret:
            # 轉換顏色供顯示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, use_container_width=True)
            
            # 自動分析
            emo, score = analyze_frame(frame)
            st.session_state.last_detected_metrics = {"emotion": emo, "score": score}
            
            # 顯示並排指標列
            m = st.session_state.last_detected_metrics
            color = EMOTION_COLOR.get(m["emotion"], "#e8e6e0")
            
            st.markdown(f"""
            <div class="metrics-container">
                <div class="metrics-item">
                    <div class="metrics-label">當前情緒</div>
                    <div class="metrics-value" style="color: {color};">{m["emotion"]}</div>
                </div>
                <div class="metrics-item" style="text-align: right;">
                    <div class="metrics-label">參與度評分</div>
                    <div class="metrics-value" style="color: white;">{m["score"]}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("無法讀取攝影機串流，請檢查權限。")

    live_metrics_panel()

    st.divider()
    
    st.subheader("📝 文法填空測驗")
    if not st.session_state.quiz_started:
        if st.button("進入測驗", type="secondary"):
            st.session_state.quiz_started = True
    
    if st.session_state.quiz_started:
        st.markdown("**Q1: DK is _____ (be) a big baby.**")
        st.text_input("Answer Q1", key="q1_input", label_visibility="collapsed")
        
        st.markdown("**Q2: I _____ (teach) for 10 years.**")
        st.text_input("Answer Q2", key="q2_input", label_visibility="collapsed")

# 底部區域：按鈕區 + AI 助手
st.divider()

# AI 助手提示區
hint_content = st.session_state.agent_hint if st.session_state.agent_hint else ""
st.markdown(f"""
<div class="hint-box">
    <span class="hint-title">🤖 AI 小助手建議：</span>
    <div class="hint-content">{hint_content}</div>
</div>
""", unsafe_allow_html=True)

# 按鈕行
b1, b2, b3, b4 = st.columns(4)
with b1:
    if st.button("▶️ 開始學習"): write_log("Start")
with b2:
    if st.button("⏸ 暫停"): write_log("Pause")
with b3:
    st.button("🤖 尋求 AI 建議", on_click=generate_hint)
with b4:
    if st.button("📤 提交作業"):
        final_score = summarize_session()
        st.success(f"作業已提交！測驗總分：{final_score}")



# with st.expander("📋 當前 Session 事件日誌預覽"):
#     if st.session_state.logs:
#         st.table(pd.DataFrame(st.session_state.logs).tail(5))
#     else:
#         st.caption("暫無紀錄")