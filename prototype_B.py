import streamlit as st
import pandas as pd
import uuid
import time
import os
import cv2
import numpy as np
from datetime import datetime
from deepface import DeepFace

# ── 頁面設定 (HCI Layout: Wide) ────────────────────────────────────────────────
st.set_page_config(
    page_title="教學實驗平台 (Version B：HCI 空間整合與低負荷版)",
    page_icon="🧠",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 環境設定與初始化 (維持 Version A 邏輯)
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

# 用於儲存特定問題的提示內容
if "hint_q1" not in st.session_state:
    st.session_state.hint_q1 = ""
if "hint_q2" not in st.session_state:
    st.session_state.hint_q2 = ""
if "reasoning_trace" not in st.session_state:
    st.session_state.reasoning_trace = []

if "last_detected_metrics" not in st.session_state:
    st.session_state.last_detected_metrics = {"emotion": "Neutral", "score": 0}
if "camera_obj" not in st.session_state:
    st.session_state.camera_obj = cv2.VideoCapture(0)

# ── Custom CSS (極簡與空間整合) ────────────────────────────────────────────────
st.markdown("""
<style>
    .stVideo { border-radius: 10px; }
    .status-label {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 5px;
    }
    /* 隱藏預設的訊息邊距，讓整合更緊密 */
    .stAlert { margin-top: -10px; margin-bottom: 10px; }
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
# 2. 核心方法 (Decision Loop & Enhanced Logging)
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
    if not st.session_state.quiz_started:
        return "Watching Video"
    q1 = st.session_state.get("q1_input", "")
    q2 = st.session_state.get("q2_input", "")
    if q1 == "":
        return "Q1"
    return "Q2"

def read_event_log():
    return round(time.time() - st.session_state.last_event_time, 2)

def write_log(event_type, hint_requested=False, score=None):
    """
    符合 Version A Schema，並支援 Version B 的精確分析 (Request_Hint_Q1/Q2)
    """
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
        "hint_type": "Action-oriented", 
        "quiz_score": score if score is not None else ""
    }
    st.session_state.logs.append(log_entry)
    st.session_state.last_event_time = time.time()
    
    df = pd.DataFrame([log_entry])
    df.to_csv(LOG_FILENAME, mode='a', header=not os.path.exists(LOG_FILENAME), index=False)

def generate_hint():
    """
    Agentic Decision Loop: 判斷目前進度並將 Hint 放置於對應問題
    """
    idle = read_event_log()
    step = get_current_step()
    q1_val = st.session_state.get("q1_input", "")
    q2_val = st.session_state.get("q2_input", "")
    
    trace = {"Observation": f"idle_time={idle}s, step={step}, input='{q1_val if step=='Q1' else q2_val}'", "Analysis": "", "Action": ""}
    
    # 清空之前的提示
    st.session_state.hint_q1 = ""
    st.session_state.hint_q2 = ""
    hint = ""

    # 條件 1 (Q1 卡住)
    if idle > 30 and step == "Q1" and q1_val == "":
        trace["Analysis"] = "學生起步困難，對 be 動詞變化不熟悉。"
        hint = "【針對 Q1 填空】 請先回放影片 0:45 處觀看規則，然後嘗試填入 be 動詞的 ing 形態。"
        st.session_state.hint_q1 = hint

    # 條件 2 (答錯)
    elif step == "Q2" and q2_val != "" and "been" not in q2_val:
        trace["Analysis"] = "漏掉現在完成進行式的關鍵 been。"
        hint = "【針對 Q2 填空】請先查看影片 1:30 的時態公式，然後將答案加上 been 與 V-ing。"
        st.session_state.hint_q2 = hint

    # 條件 3 (預設)
    else:
        trace["Analysis"] = "正常行為模式。"
        hint = "請繼續觀看影片，並根據重點嘗試作答下一題。作答完畢請提交作業。"
        if step == "Q1": st.session_state.hint_q1 = hint
        else: st.session_state.hint_q2 = hint

    trace["Action"] = hint
    st.session_state.reasoning_trace.append(trace)

def summarize_session():
    q1 = st.session_state.get("q1_input", "").lower()
    q2 = st.session_state.get("q2_input", "").lower()
    score = 0
    if "being" in q1: score += 50
    if "been" in q2 and ("teaching" in q2 or "taught" in q2): score += 50
    write_log("Submit", score=score)
    return score

# ══════════════════════════════════════════════════════════════════════════════
# 3. UI 介面重構 (Version B：空間整合)
# ══════════════════════════════════════════════════════════════════════════════

# 頂部導航
st.title("🧠 教學實驗平台 (Version B：HCI 空間整合與低負荷版)")

# 主分欄
col_left, col_right = st.columns([2, 1], gap="large")

# --- 左欄：環境與內容 ---
with col_left:
    l_top1, l_top2 = st.columns(2)
    with l_top1:
        st.session_state.student_id = st.text_input("學號 Student ID", placeholder="B11234567")
    with l_top2:
        st.session_state.env_mode = st.radio("實驗環境", ["Classroom", "Home"], horizontal=True)
    
    st.video("https://www.youtube.com/watch?v=PIZSp-yNShk")
    
    # 控制項下移，保持影片焦點
    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        if st.button("▶️ 開始學習"): write_log("Start")
    with b_col2:
        if st.button("⏸ 暫停"): write_log("Pause")
    with b_col3:
        if st.button("📤 提交作業"):
            final_score = summarize_session()
            st.success(f"作業提交成功！得分：{final_score}")

# --- 右欄：監測與測驗 (核心空間整合區) ---
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
    
    st.subheader("📝 文法填空測驗")
    if not st.session_state.quiz_started:
        if st.button("進入測驗", use_container_width=True):
            st.session_state.quiz_started = True
            write_log("Enter_Quiz")
            st.rerun()
    
    if st.session_state.quiz_started:
        # --- Q1 區塊 ---
        st.markdown('<div class="status-label">📝 作答中</div>', unsafe_allow_html=True)
        st.markdown("**Q1: DK is _____ (be) a big baby.**")
        st.text_input("Answer Q1", key="q1_input", label_visibility="collapsed")
        
        # Proximal Hint Integration for Q1
        if st.session_state.hint_q1:
            st.markdown('<div class="status-label">💡 AI 引導中</div>', unsafe_allow_html=True)
            st.info(st.session_state.hint_q1)
            if st.button("📺 帶我去回看 0:45 處", key="action_q1"):
                st.toast("已為您標記重點段落：0:45", icon="📌")

        st.markdown("---")

        # --- Q2 區塊 ---
        st.markdown('<div class="status-label">📝 作答中</div>', unsafe_allow_html=True)
        st.markdown("**Q2: I _____ (teach) for 10 years.**")
        st.text_input("Answer Q2", key="q2_input", label_visibility="collapsed")
        
        # Proximal Hint Integration for Q2
        if st.session_state.hint_q2:
            st.markdown('<div class="status-label">💡 AI 引導中</div>', unsafe_allow_html=True)
            st.info(st.session_state.hint_q2)
            if st.button("📺 帶我去回看 1:30 處", key="action_q2"):
                st.toast("已為您標記重點段落：1:30", icon="📌")

    st.divider()
    # 尋求建議按鈕就在測驗正下方，縮短路徑
    st.button("🤖 尋求 AI 建議", on_click=generate_hint, use_container_width=True)


# with st.expander("📋 當前 Session 事件日誌預覽"):
#     if st.session_state.logs:
#         st.table(pd.DataFrame(st.session_state.logs).tail(5))
#     else:
#         st.caption("暫無紀錄")