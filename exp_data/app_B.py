import streamlit as st
import pandas as pd
import time
import os

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="Trace Viewer - Condition B", layout="wide")

IMG_DIR = "images/"
XAI_DIR = "xAI_images/"
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
# 讀取資料（強化防呆版）
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")

    # 🔥 清掉欄位空白
    df.columns = df.columns.str.strip()

    required_cols = [
        "id", "image", "prediction", "confidence",
        "image_gradcam", "image_shap"
    ]

    # 🔥 檢查缺欄位
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"CSV 缺少欄位：{missing}")
        st.write("目前欄位：", df.columns)
        st.stop()

    # 🔥 填補 NaN（避免 join 爆炸）
    df["image"] = df["image"].fillna("")
    df["image_gradcam"] = df["image_gradcam"].fillna("")
    df["image_shap"] = df["image_shap"].fillna("")

    keep_cols = required_cols.copy()
    if "ground_truth" in df.columns:
        keep_cols.append("ground_truth")

    return df[keep_cols]

df = load_data()

# =========================
# 標題
# =========================
st.title("情緒辨識 AI 輔助系統 (Condition B - 具備解釋資訊)")

# =========================
# 結束畫面
# =========================
if st.session_state.current_step >= TOTAL_QUESTIONS:
    st.success("實驗結束！")

    results_df = pd.DataFrame(st.session_state.results)
    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="下載實驗結果 (results_B.csv)",
        data=csv,
        file_name="results_B.csv",
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
# 工具函式（避免重複寫）
# =========================
def show_image_safe(base_dir, file_name, label):
    if not isinstance(file_name, str) or file_name.strip() == "":
        st.warning(f"{label}：無資料")
        return

    path = os.path.join(base_dir, file_name)

    if os.path.exists(path):
        st.image(path, width=300)
    else:
        st.error(f"{label}：找不到檔案\n{path}")

# =========================
# 三欄對比（核心）
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### (A) 原始臉部表情")
    show_image_safe(IMG_DIR, row["image"], "原始圖片")

with col2:
    st.markdown("### (B) 模型關注區域 (Grad-CAM)")
    show_image_safe(XAI_DIR, row["image_gradcam"], "Grad-CAM")

with col3:
    st.markdown("### (C) 特徵貢獻分析 (SHAP)")
    show_image_safe(XAI_DIR, row["image_shap"], "SHAP")

# =========================
# AI 診斷報告
# =========================
st.markdown("---")
st.markdown(f"## AI 預測結果：{row['prediction']}")
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

    is_correct = None
    if "ground_truth" in df.columns:
        is_correct = row["prediction"] == row["ground_truth"]

    st.session_state.results.append({
        "id": row["id"],
        "user_confidence": user_confidence,
        "is_correct": is_correct,
        "response_time": reaction_time
    })

    # reset timer
    st.session_state.start_time = time.time()

    # 下一題
    st.session_state.current_step += 1
    st.rerun()