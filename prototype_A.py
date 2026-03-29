import streamlit as st
import pandas as pd
import uuid
import time
import os
from datetime import datetime

# ── 頁面設定 ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="教學實驗平台 (Version A：行動導向提示版)",
    page_icon="🎓",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. 環境設定與初始化 (Env & Session)
# ══════════════════════════════════════════════════════════════════════════════
LOG_FILENAME = "study_logs_A.csv"

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
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2. 核心方法 (Decision Loop & Logging)
# ══════════════════════════════════════════════════════════════════════════════

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
    
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "student_id": st.session_state.get("student_id", "Unknown"),
        "session_id": st.session_state.session_id,
        "environment": st.session_state.get("env_mode", "Classroom"),
        "event_type": event_type,
        "current_step": current_step,
        "idle_time": idle,
        "hint_requested": hint_requested,
        "hint_type": "Action-oriented",
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
        trace["Analysis"] = "學生起步困難，對 be 動詞變化不熟悉。"
        hint = "【針對 Q1 填空】 請先回放影片 0:45 處觀看規則，然後嘗試填入 be 動詞的 ing 形態。"
    
    # 條件 2 (答錯)
    elif step == "Q2" and q2_val != "" and "been" not in q2_val:
        trace["Analysis"] = "漏掉現在完成進行式的關鍵 been。"
        hint = "【針對 Q2 填空】請先查看影片 1:30 的時態公式，然後將答案加上 been 與 V-ing。"
    
    # 條件 3 (預設)
    else:
        trace["Analysis"] = "正常行為模式。"
        hint = "請繼續觀看影片，並根據重點嘗試作答下一題。作答完畢請提交作業。"

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
st.title("🎓 教學實驗平台 (Version A：行動導向提示版)")
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
    st.camera_input("請對準鏡頭", key="camera_input")
    
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
