import os, time, math, warnings, random, io
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler

# =========================
# Page config & theming
# =========================
st.set_page_config(page_title="Gold Price LSTM Dashboard", layout="wide")
st.title("📈 Dự báo giá vàng với LSTM — Đồ án KHDL")
st.caption("Tương tác: tải dữ liệu, cấu hình tham số, huấn luyện, xem biểu đồ & tải kết quả.")

# Reproducibility
SEED = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# =========================
# Sidebar — cấu hình
# =========================
st.sidebar.header("⚙️ Cấu hình")
DEFAULT_PATH = "/content/drive/MyDrive/Đồ án/Đồ án chuyên ngành KHDL/Gold Price (2013-2023).csv"

uploaded = st.sidebar.file_uploader("Tải CSV (tùy chọn)", type=["csv"]) 
use_default = st.sidebar.checkbox("Dùng đường dẫn mặc định", value=False)
custom_path = st.sidebar.text_input("hoặc nhập đường dẫn CSV", value="")

DATE_COL = st.sidebar.text_input("Tên cột ngày", value="Date")
PRICE_COL = st.sidebar.text_input("Tên cột giá", value="Price")
TRAIN_END = st.sidebar.text_input("Mốc chia Train/Test (YYYY-MM-DD)", value="2021-12-31")
LOOKBACK   = st.sidebar.number_input("LOOKBACK (cửa sổ chuỗi)", min_value=10, max_value=365, value=60, step=5)
EPOCHS     = st.sidebar.number_input("EPOCHS", min_value=1, max_value=500, value=120, step=5)
BATCH      = st.sidebar.number_input("BATCH SIZE", min_value=1, max_value=512, value=32, step=1)
VERBOSE    = st.sidebar.selectbox("Verbose", options=[0,1,2], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Dự báo tương lai")
PRESET = st.sidebar.selectbox("Chọn nhanh", options=["5 ngày","10 ngày","1 tháng (30)","Tùy chỉnh"], index=0)
if PRESET == "5 ngày":
    FORECAST_DAYS = 5
elif PRESET == "10 ngày":
    FORECAST_DAYS = 10
elif PRESET == "1 tháng (30)":
    FORECAST_DAYS = 30
else:
    FORECAST_DAYS = st.sidebar.number_input("Số ngày dự báo", min_value=1, max_value=365, value=7, step=1)

PLOT_START_YEAR = st.sidebar.number_input("Hiển thị biểu đồ từ năm", min_value=2000, max_value=2100, value=2020, step=1)

col_btn1, col_btn2 = st.sidebar.columns(2)
train_btn = col_btn1.button("🚀 Train")
reset_btn = col_btn2.button("🔄 Reset")

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Helper functions
# =========================

def clean_numeric_cell(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().replace("$","").replace("€","").replace("%","").replace("K","000")
    s = s.replace(",", "")
    if s in ["", "-", "null", "None", "NaN", "—", "–"]:
        return np.nan
    try:
        return float(s)
    except Exception:
        s2 = s.replace(".", "").replace(",", ".")
        try:
            return float(s2)
        except Exception:
            return np.nan

@st.cache_data(show_spinner=False)
def load_data(file_obj_or_path, date_col, price_col):
    if file_obj_or_path is None:
        raise ValueError("Chưa cung cấp dữ liệu CSV.")
    df = pd.read_csv(file_obj_or_path)
    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{date_col}' hoặc '{price_col}' trong CSV.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = df[price_col].apply(clean_numeric_cell).astype(float)
    data = df[[date_col, price_col]].dropna().sort_values(date_col).reset_index(drop=True)
    data = data.rename(columns={date_col: "date", price_col: "close"})
    return data

# ... (giữ nguyên các hàm khác)

# Trong phần plot forecast:
# Thay vì hiển thị toàn bộ lịch sử, chỉ hiển thị từ năm được chọn

                figf, axf = plt.subplots(figsize=(12,3.5))
                base_all = st.session_state.data.set_index("date")["close"]
start_dt = pd.Timestamp(f"{int(SHOW_FROM_YEAR)}-01-01")
base = base_all.loc[start_dt:]
                base = base[base.index.year >= PLOT_START_YEAR]
                axf.plot(base.index, base.values, color="black", label="Historical")
                axf.plot(fdf["date"], fdf["forecast"], color="green", label=f"Forecast +{len(fdf)}d")
                axf.set_title("Future Forecast (recursive)")
                axf.set_xlabel("Date"); axf.set_ylabel("Price")
                axf.legend(); figf.tight_layout()
                st.pyplot(figf, use_container_width=True)
