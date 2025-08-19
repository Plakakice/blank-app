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

col_btn1, col_btn2 = st.sidebar.columns(2)
train_btn = col_btn1.button("🚀 Train")
reset_btn = col_btn2.button("🔄 Reset")

SAVE_DIR = "outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Helper functions
# =========================

def clean_numeric_cell(s):
    """'1,826.20' -> 1826.20; '—' -> NaN"""
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

@st.cache_data(show_spinner=False)
def make_sequences(data_df, lookback, train_end):
    from collections import deque
    sc_close = MinMaxScaler()
    train = data_df[data_df["date"] <= pd.to_datetime(train_end)].copy()
    sc_close.fit(train[["close"]].values)
    close_all_scaled = pd.Series(
        sc_close.transform(data_df[["close"]].values).ravel(),
        index=data_df["date"], name="close_scaled")
    y_scaled = close_all_scaled.shift(-1)
    Xs, ys, idx = [], [], []
    dq = deque(maxlen=lookback)
    for t, v in close_all_scaled.items():
        dq.append(v)
        if len(dq) == lookback:
            yv = y_scaled.loc[t]
            if np.isnan(yv):
                break
            Xs.append(np.array(dq))
            ys.append(yv)
            idx.append(t)
    X_all = np.array(Xs)
    y_all = np.array(ys)
    idx_all = pd.DatetimeIndex(idx)
    mask_train = idx_all <= pd.to_datetime(train_end)
    X_train, y_train = X_all[mask_train], y_all[mask_train]
    X_test,  y_test  = X_all[~mask_train], y_all[~mask_train]
    idx_test = idx_all[~mask_train]
    X_train = X_train.reshape(-1, lookback, 1)
    X_test  = X_test.reshape(-1, lookback, 1)
    return sc_close, (X_train, y_train), (X_test, y_test), idx_test


def build_lstm_model(lookback):
    model = models.Sequential([
        layers.Input(shape=(lookback, 1)),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def plot_performance(data_df, train_end, idx_test, y_pred, title="Model Performance on Gold Price Prediction (LSTM)"):
    close = data_df.set_index("date")["close"]
    fig, ax = plt.subplots(figsize=(12,4))
    ax.set_facecolor("#fff59d")
    ax.plot(close.loc[:train_end].index, close.loc[:train_end], color="black", label="Training Data")
    ax.plot(close.loc[train_end:].index, close.loc[train_end:], color="blue",  label="Actual Test Data")
    ax.plot(idx_test, y_pred, color="red", label="Predicted Test Data")
    ax.set_title(title)
    ax.set_xlabel("Date"); ax.set_ylabel("Price")
    ax.legend(); fig.tight_layout()
    return fig

# =========================
# Main app flow
# =========================

# Resolve data source
csv_source = None
if uploaded is not None:
    csv_source = uploaded
elif use_default:
    if os.path.exists(DEFAULT_PATH):
        csv_source = DEFAULT_PATH
    else:
        st.warning("Không tìm thấy file mặc định. Hãy tải CSV hoặc nhập đường dẫn hợp lệ.")
elif custom_path:
    if os.path.exists(custom_path):
        csv_source = custom_path
    else:
        st.warning("Đường dẫn CSV không tồn tại.")

# Session state
if reset_btn:
    for k in ["data", "scaler", "X_train", "y_train", "X_test", "y_test", "idx_test", "model", "history", "metrics", "pred_df"]:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# 1) Load & preview
if csv_source is not None and "data" not in st.session_state:
    with st.spinner("Đang tải & tiền xử lý dữ liệu…"):
        try:
            data = load_data(csv_source, DATE_COL, PRICE_COL)
            st.session_state.data = data
        except Exception as e:
            st.error(f"Lỗi dữ liệu: {e}")

if "data" in st.session_state:
    data = st.session_state.data
    c1, c2 = st.columns([2,1])
    with c1:
        st.subheader("👀 Mẫu dữ liệu")
        st.dataframe(data.head(10), use_container_width=True)
    with c2:
        st.metric("Số dòng", len(data))
        st.write("Khoảng thời gian:", data["date"].min().date(), "→", data["date"].max().date())

    # 2) Make sequences
    if "X_train" not in st.session_state:
        with st.spinner("Đang tạo chuỗi (sequences)…"):
            sc_close, (X_train, y_train), (X_test, y_test), idx_test = make_sequences(data, LOOKBACK, TRAIN_END)
            st.session_state.scaler = sc_close
            st.session_state.X_train, st.session_state.y_train = X_train, y_train
            st.session_state.X_test,  st.session_state.y_test  = X_test, y_test
            st.session_state.idx_test = idx_test

    # 3) Train
    if train_btn:
        X_train = st.session_state.X_train; y_train = st.session_state.y_train
        X_test  = st.session_state.X_test;  y_test  = st.session_state.y_test
        if X_train.size == 0 or X_test.size == 0:
            st.error("Không đủ dữ liệu train/test. Hãy kiểm tra TRAIN_END hoặc LOOKBACK.")
        else:
            cb_list = [
                callbacks.ReduceLROnPlateau(factor=0.5, patience=5, verbose=1, monitor="val_loss"),
                callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss", verbose=1),
            ]
            model = build_lstm_model(LOOKBACK)
            t0 = time.perf_counter()
            with st.spinner("Đang huấn luyện mô hình…"):
                hist = model.fit(
                    X_train, y_train,
                    validation_split=0.15,
                    epochs=EPOCHS,
                    batch_size=BATCH,
                    verbose=VERBOSE,
                    callbacks=cb_list   )
            train_time = time.perf_counter() - t0
            st.session_state.model = model
            st.session_state.history = hist.history

            # 4) Predict & inverse transform
            y_pred_scaled = model.predict(st.session_state.X_test, verbose=0).ravel()
            y_pred = st.session_state.scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
            y_true = st.session_state.scaler.inverse_transform(st.session_state.y_test.reshape(-1,1)).ravel()

            # 5) Metrics
            rmse = math.sqrt(np.mean((y_true - y_pred)**2))
            mae  = np.mean(np.abs(y_true - y_pred))
            mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-9))) * 100
            metrics = {"RMSE": float(rmse), "MAE": float(mae), "MAPE_%": float(mape), "Train_Time_s": round(train_time,1)}
            st.session_state.metrics = metrics

            # 6) Pred DataFrame
            pred_df = pd.DataFrame({
                "date": st.session_state.idx_test,
                "y_true": y_true,
                "y_pred": y_pred
            })
            st.session_state.pred_df = pred_df

    # 4) Show results
    if "metrics" in st.session_state and "pred_df" in st.session_state:
        m = st.session_state.metrics
        pred_df = st.session_state.pred_df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{m['RMSE']:.2f}")
        c2.metric("MAE", f"{m['MAE']:.2f}")
        c3.metric("MAPE", f"{m['MAPE_%']:.2f}%")
        c4.metric("Thời gian train", f"{m['Train_Time_s']:.1f}s")

        # Plot
        fig = plot_performance(st.session_state.data, TRAIN_END, st.session_state.idx_test, pred_df["y_pred"].values)
        st.pyplot(fig, use_container_width=True)

        # History chart (loss)
        if "history" in st.session_state:
            hist = st.session_state.history
            if "loss" in hist and "val_loss" in hist:
                fig2, ax2 = plt.subplots(figsize=(12,3))
                ax2.plot(hist["loss"], label="loss")
                ax2.plot(hist["val_loss"], label="val_loss")
                ax2.set_title("Learning Curve"); ax2.set_xlabel("epoch"); ax2.set_ylabel("MSE")
                ax2.legend(); fig2.tight_layout()
                st.pyplot(fig2, use_container_width=True)

        # Downloads
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="⬇️ Tải metrics.json",
                data=io.BytesIO(pd.Series(m).to_json().encode()),
                file_name="metrics.json",
                mime="application/json"
            )
        with col_b:
            csv_bytes = pred_df.to_csv(index=False).encode()
            st.download_button(
                label="⬇️ Tải predictions.csv",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )

        # Save model (optional)
        with st.expander("💾 Lưu/Load mô hình (tùy chọn)"):
            save_col, load_col = st.columns(2)
            if save_col.button("Lưu mô hình vào outputs/model.keras"):
                try:
                    st.session_state.model.save(os.path.join(SAVE_DIR, "model.keras"))
                    st.success("Đã lưu mô hình.")
                except Exception as e:
                    st.error(f"Không thể lưu mô hình: {e}")
            uploaded_weights = load_col.file_uploader("Tải file model.keras để load lại", type=["keras"], key="uploader_model")
            if uploaded_weights is not None and st.button("Load mô hình đã tải"):
                try:
                    with open(os.path.join(SAVE_DIR, "tmp_model.keras"), "wb") as f:
                        f.write(uploaded_weights.read())
                    st.session_state.model = tf.keras.models.load_model(os.path.join(SAVE_DIR, "tmp_model.keras"))
                    st.success("Đã load mô hình.")
                except Exception as e:
                    st.error(f"Không thể load mô hình: {e}")

else:
    st.info("➡️ Hãy tải CSV, chọn dùng file mặc định, hoặc nhập đường dẫn rồi nhấn **Train**.")

st.markdown("---")
st.caption("Gợi ý: Deploy dễ dàng với Streamlit Community Cloud hoặc Hugging Face Spaces. Nếu chạy trong Google Colab, ưu tiên dùng Gradio thay vì Streamlit.")
