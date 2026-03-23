import streamlit as st
import pandas as pd
import time
import os

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="Trace Viewer - Condition A", layout="centered")

IMG_DIR = "images/"  # 圖片資料夾

# =========================
# 初始化 Session State
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
    # 只保留 Condition A 允許欄位
    return df[["id", "image", "prediction", "confidence"]]

df = load_data()

TOTAL = 30

# =========================
# UI 標題
# =========================
st.title("情緒辨識 AI 輔助系統 (Condition A)")

# =========================
# 若已完成
# =========================
if st.session_state.current_step >= TOTAL:
    st.success("實驗結束，感謝您的參與！")

    result_df = pd.DataFrame(st.session_state.results)

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="下載實驗結果 (results_A.csv)",
        data=csv,
        file_name="results_A.csv",
        mime="text/csv"
    )

    st.stop()

# =========================
# 進度顯示
# =========================
progress = st.session_state.current_step / TOTAL
st.progress(progress)

st.write(f"目前第 {st.session_state.current_step + 1} / {TOTAL} 題")

# =========================
# 取得當前題目
# =========================
row = df.iloc[st.session_state.current_step]

# =========================
# 顯示圖片
# =========================
img_path = os.path.join(IMG_DIR, row["image"])

if os.path.exists(img_path):
    st.image(img_path, use_container_width=True)
else:
    st.warning(f"找不到圖片：{img_path}")

# =========================
# AI 預測顯示
# =========================
st.markdown("### 🤖 模型預測結果")

st.markdown(
    f"<h2 style='color:black;'>模型預測結果：{row['prediction']}</h2>",
    unsafe_allow_html=True
)

st.markdown(
    f"<h4 style='color:gray;'>信心度：{row['confidence']:.2f}</h4>",
    unsafe_allow_html=True
)

st.divider()

# =========================
# 使用者決策區
# =========================
st.markdown("### 🧠 您的判斷")

emotion = st.radio(
    "Q1：您的最終情緒判斷？",
    ["困惑", "挫折", "無聊", "喜悅", "驚訝", "投入"],
    index=None
)

confidence = st.slider(
    "Q2：您對此判斷的信心度？",
    min_value=1,
    max_value=5,
    value=3
)

# =========================
# 提交按鈕
# =========================
if st.button("提交判斷並下一張"):

    if emotion is None:
        st.warning("請先選擇情緒！")
    else:
        end_time = time.time()
        reaction_time = end_time - st.session_state.start_time

        # ⚠️ Condition A 不顯示 ground truth
        # 若 CSV 有 ground_truth 可用於計算（不顯示）
        is_correct = None
        if "ground_truth" in df.columns:
            is_correct = int(emotion == df.iloc[st.session_state.current_step]["ground_truth"])

        # 紀錄結果
        st.session_state.results.append({
            "id": row["id"],
            "user_emotion": emotion,
            "user_confidence": confidence,
            "reaction_time": reaction_time,
            "is_correct": is_correct
        })

        # 重置時間
        st.session_state.start_time = time.time()

        # 下一題
        st.session_state.current_step += 1

        st.rerun()