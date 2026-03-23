import streamlit as st
import pandas as pd
import time
import os

# =========================
# 基本設定
# =========================
st.set_page_config(page_title="Trace Viewer - Condition C", layout="wide")

ORIG_DIR = "images/"
XAI_DIR = "xAI_images/"
TOTAL_QUESTIONS = 30
IMG_WIDTH = 280  # 固定圖片大小（避免實驗偏差）

# =========================
# Session State 初始化
# =========================
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "results" not in st.session_state:
    st.session_state.results = []

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

# =========================
# 讀取資料（強化防呆）
# =========================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data.csv")
    except:
        df = pd.read_csv("data.csv", engine="python", on_bad_lines="skip")

    df.columns = df.columns.str.strip()

    required_cols = [
        "id", "image", "ground_truth", "prediction", "confidence",
        "image_gradcam", "image_shap", "why_not_text", "risk_level"
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"CSV 缺少欄位：{missing}")
        st.write("目前欄位：", df.columns)
        st.stop()

    # 避免 NaN 問題
    for col in ["image", "image_gradcam", "image_shap", "why_not_text"]:
        df[col] = df[col].fillna("")

    return df[required_cols]

df = load_data()

# =========================
# 標題
# =========================
st.title("情緒辨識 AI 輔助系統 (Condition C - 信任校準介面)")

# =========================
# 結束畫面
# =========================
if st.session_state.current_step >= TOTAL_QUESTIONS:
    st.success("實驗完成，感謝您的參與！")

    results_df = pd.DataFrame(st.session_state.results)
    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="下載實驗結果 (results_C.csv)",
        data=csv,
        file_name="results_C.csv",
        mime="text/csv"
    )

    st.stop()

# =========================
# 當前題目
# =========================
row = df.iloc[st.session_state.current_step]

# =========================
# 進度條
# =========================
progress = (st.session_state.current_step + 1) / TOTAL_QUESTIONS
st.progress(progress)
st.write(f"第 {st.session_state.current_step + 1} / {TOTAL_QUESTIONS} 題")

# =========================
# 🔥 風險警示區（核心）
# =========================
risk = str(row["risk_level"]).lower()

if risk == "high":
    st.error("⚠️ 系統警告：此判斷之特徵依據可能存在嚴重偏差，建議加強人工審核。")
elif risk == "low":
    st.success("系統提示：特徵依據符合典型情緒模型。")
else:
    st.info("風險資訊不足")

# =========================
# 圖片顯示工具（固定大小 + 防炸）
# =========================
def show_image(base_dir, file_name, label):
    if not isinstance(file_name, str) or file_name.strip() == "":
        st.warning(f"{label}：無資料")
        return

    path = os.path.join(base_dir, file_name)

    if os.path.exists(path):
        st.image(path, width=IMG_WIDTH)
    else:
        st.error(f"{label}：找不到\n{path}")

# =========================
# 視覺證據區（三欄）
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### (A) 原始臉部表情")
    show_image(ORIG_DIR, row["image"], "原始圖片")

with col2:
    st.markdown("### (B) AI 關注區域 (Grad-CAM)")
    show_image(XAI_DIR, row["image_gradcam"], "Grad-CAM")

with col3:
    st.markdown("### (C) 特徵貢獻分析 (SHAP)")
    show_image(XAI_DIR, row["image_shap"], "SHAP")

# =========================
# 診斷報告
# =========================
st.markdown("---")
st.markdown(f"## AI 預測結果：{row['prediction']}")
st.markdown(f"### 信心度：{row['confidence']}")

# 🔥 Why Not（反事實）
if row["why_not_text"].strip() != "":
    st.warning(f"🔍 Why Not？\n{row['why_not_text']}")
else:
    st.info("無反事實說明")

# =========================
# 使用者互動（核心）
# =========================
st.markdown("---")

user_confidence = st.slider(
    "Q1：您對此判斷的信心度？",
    min_value=1,
    max_value=5,
    value=3
)

user_adoption = st.radio(
    "Q2：您是否參考並採納了 AI 的建議？",
    ["完全採納", "部分參考", "完全不採納"]
)

# =========================
# 提交按鈕
# =========================
if st.button("確認並提交下一張"):

    end_time = time.time()
    response_time = end_time - st.session_state.start_time

    st.session_state.results.append({
        "id": row["id"],
        "ground_truth": row["ground_truth"],
        "prediction": row["prediction"],
        "risk_level": row["risk_level"],
        "user_confidence": user_confidence,
        "user_adoption": user_adoption,
        "response_time": response_time
    })

    # 重置計時
    st.session_state.start_time = time.time()

    # 下一題
    st.session_state.current_step += 1
    st.rerun()