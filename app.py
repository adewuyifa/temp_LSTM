# app.py
import streamlit as st
import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import plotly.graph_objects as go
from model import LSTMForecaster

# ============================================================
# CONFIGURATION & STYLE
# ============================================================
st.set_page_config(page_title="Cairo Weather Forecast", page_icon="🌤️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f7f9fc; padding: 2rem;}
    .title {text-align: center; color: #1e3d59; font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;}
    .subtitle {text-align: center; color: #555; font-size: 1rem; margin-bottom: 2rem;}
    .section {background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & SCALER
# ============================================================
device = torch.device("cpu")

@st.cache_resource
def load_model_and_scaler():
    scaler = joblib.load("scaler.save")
    model = LSTMForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load("cairo_lstm_torch.pt", map_location=device))
    model.eval()
    return model, scaler

model, scaler = load_model_and_scaler()

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv("Cairo-Weather.csv", parse_dates=['date'])
data = df[['temperature_2m_mean']].dropna()
scaled = scaler.transform(data)

# ============================================================
# FORECAST FUNCTION
# ============================================================
def forecast_future(model, data, lookback=30, steps=7):
    model.eval()
    seq = data[-lookback:].reshape(1, lookback, 1)
    seq = torch.tensor(seq, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            y_pred = model(seq).numpy().flatten()
            preds.append(y_pred[-1])
            seq = torch.tensor(
                np.append(seq.numpy().flatten(), y_pred[-1])[-lookback:].reshape(1, lookback, 1),
                dtype=torch.float32
            )
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1))

# ============================================================
# PAGE CONTENT
# ============================================================
st.markdown('<div class="title">🌤️ Cairo Temperature Forecast Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">LSTM-based daily temperature forecasting for Cairo, Egypt</div>', unsafe_allow_html=True)

# ============ Layout Containers ============
with st.container():
    col1, col2 = st.columns([1.2, 1])

    # ---------------------------------------
    # LEFT COLUMN → INPUTS + FORECAST TABLE
    # ---------------------------------------
    with col1:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("🗓️ Select Forecast Range")

        start_date = st.date_input(
            "Start Date",
            min_value=df['date'].iloc[-1].date() + pd.Timedelta(days=1),
            value=df['date'].iloc[-1].date() + pd.Timedelta(days=1)
        )

        end_date = st.date_input(
            "End Date",
            min_value=start_date,
            value=start_date + pd.Timedelta(days=7)
        )

        num_days = (end_date - start_date).days + 1
        st.write(f"📆 Forecasting **{num_days} days** ahead ({start_date} → {end_date})")

        # ---------------------------------------
        # FORECAST BUTTON
        # ---------------------------------------
        if st.button("🔮 Generate Forecast"):
            forecast_values = forecast_future(model, scaled, lookback=30, steps=num_days)
            forecast_dates = pd.date_range(start=start_date, periods=num_days)

            # DataFrame
            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Forecast (°C)": forecast_values.flatten()
            })

            st.markdown("### 🔮 Forecast Results")
            st.dataframe(forecast_df.style.format({"Forecast (°C)": "{:.2f}"}), use_container_width=True)
        else:
            forecast_values = None
            forecast_dates = None

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------
    # RIGHT COLUMN → PLOTLY VISUALIZATION
    # ---------------------------------------
    with col2:
        st.markdown('<div class="section">', unsafe_allow_html=True)
        st.subheader("📈 Temperature Trends")

        # Historical tail
        history_tail = df.tail(60)

        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=history_tail['date'],
            y=history_tail['temperature_2m_mean'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='#1e88e5'),
            marker=dict(size=4)
        ))

        # Forecast line (only if forecast exists)
        if forecast_values is not None and forecast_dates is not None:
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_values.flatten(),
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#e53935', dash='dash'),
                marker=dict(size=6)
            ))

        # Layout
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Temperature (°C)',
            title='Cairo Temperature Forecast',
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig.update_xaxes(showgrid=True, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridcolor='lightgray')

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Developed with ❤️ using PyTorch + Streamlit + Plotly | Data: Cairo Weather and Climate Dataset")
