import streamlit as st
import pandas as pd
import time
import os

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="Trace Viewer - Condition A", layout="centered")

IMG_DIR = "images/"
TOTAL_QUESTIONS = 30

# =========================
# 初始化 session state
# =========================
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "results" not in st.session_state:
    st.session_state.results = []

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# =========================
# 讀取資料
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")
    
    # 僅保留必要欄位（Condition A 限制）
    required_cols = ["id", "image", "prediction", "confidence"]
    df = df[required_cols + ([col for col in df.columns if col == "ground_truth"])]
    
    return df

df = load_data()

# =========================
# 標題
# =========================
st.title("情緒辨識 AI 輔助系統 (Condition A)")

# =========================
# 結束畫面
# =========================
if st.session_state.current_step >= TOTAL_QUESTIONS:
    st.success("實驗結束！")

    results_df = pd.DataFrame(st.session_state.results)

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="下載實驗結果 (results_A.csv)",
        data=csv,
        file_name="results_A.csv",
        mime="text/csv"
    )

    st.stop()

# =========================
# 目前題目
# =========================
row = df.iloc[st.session_state.current_step]

# =========================
# 進度條
# =========================
progress = (st.session_state.current_step + 1) / TOTAL_QUESTIONS
st.progress(progress)
st.write(f"目前第 {st.session_state.current_step + 1} / {TOTAL_QUESTIONS} 題")

# =========================
# 顯示圖片
# =========================
img_path = os.path.join(IMG_DIR, row["image"])

if os.path.exists(img_path):
    st.image(img_path, use_container_width=True)
else:
    st.warning(f"找不到圖片：{img_path}")

# =========================
# AI 預測區
# =========================
st.markdown(f"## 模型預測結果：{row['prediction']}")
st.markdown(f"### 信心度：{row['confidence']}")

# =========================
# 使用者信心度
# =========================
user_confidence = st.slider(
    "您對此判斷的信心度？",
    min_value=1,
    max_value=5,
    value=3
)

# =========================
# 提交按鈕
# =========================
if st.button("提交判斷並下一張"):

    end_time = time.time()
    reaction_time = end_time - st.session_state.start_time

    # 是否與 ground_truth 一致（若存在）
    is_correct = None
    if "ground_truth" in df.columns:
        is_correct = row["prediction"] == row["ground_truth"]

    # 紀錄
    st.session_state.results.append({
        "id": row["id"],
        "user_confidence": user_confidence,
        "is_correct": is_correct,
        "reaction_time": reaction_time
    })

    # 重設時間
    st.session_state.start_time = time.time()

    # 下一題
    st.session_state.current_step += 1

    st.rerun()