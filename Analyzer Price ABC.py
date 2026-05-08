import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from plotly.subplots import make_subplots

# =============================
# UI CONFIG
# =============================
st.set_page_config(layout="wide", page_title="ABC Trend")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 18px !important; margin-bottom: 10px !important; }
        .flag-label { font-size: 8px; color: #888; margin-bottom: -5px; text-transform: uppercase; font-weight: 600; }
        .flag-value { font-size: 13px; color: #fff; white-space: nowrap; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>📅 Price ABC Multi-TimeFrame Comparison</h1>", unsafe_allow_html=True)

NY_TZ = ZoneInfo("America/New_York")
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

col_input, col_refresh = st.columns([1, 1])

with col_input:
    st.markdown(f"<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>Enter Ticker | Last Ref: {st.session_state.last_refresh.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)
    ticker = st.text_input("Ticker", value="APP", label_visibility="collapsed").strip().upper()

with col_refresh:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("🔄"):
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.cache_data.clear()
        st.rerun()

# =============================
# CORE ANALYSIS ENGINE
# =============================
def get_and_analyze(ticker, period_str):
    settings = {
        "1mo": {"yf_period": "1mo", "cci": 7, "gap": 3},
        "3mo": {"yf_period": "3mo", "cci": 14, "gap": 3},
        "6mo": {"yf_period": "6mo", "cci": 20, "gap": 5}
    }
    s = settings[period_str]
    
    # 1. FETCH HISTORICAL DATA
    # Fetching extra 5 days of data to ensure the 5-day rolling window is primed at the start of the chart
    df = yf.download(ticker, period=s["yf_period"], interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
    close, high, low, volume = df["Close"].squeeze(), df["High"].squeeze(), df["Low"].squeeze(), df["Volume"].squeeze()

    # 2. FETCH LIVE PRICE (Intraday + Pre/Post)
    live_data = yf.download(ticker, period="1d", interval="15m", prepost=True, auto_adjust=True, progress=False)
    if not live_data.empty:
        if isinstance(live_data.columns, pd.MultiIndex):
            live_data.columns = live_data.columns.get_level_values(0)
        price_now = float(live_data["Close"].iloc[-1])
    else:
        price_now = float(close.iloc[-1])

    # 3. ROLLING 5-DAY VWAP (Updated Logic)
    pv = (close * volume).rolling(window=5).sum()
    vol_sum = volume.rolling(window=5).sum().replace(0, np.nan)
    vwap = pv / vol_sum

    # ATR
    tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    atr_val = tr.rolling(14).mean().mean()

    # CCI
    tp = (high + low + close) / 3
    sma = tp.rolling(s["cci"]).mean()
    mad = tp.rolling(s["cci"]).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma) / (0.015 * mad)
    
    # Pivot Logic
    pivots = []
    last_p, last_idx = close.iloc[0], 0
    for i in range(2, len(close)-2):
        if i - last_idx < s["gap"]: continue
        is_h = close.iloc[i] > close.iloc[i-1] and close.iloc[i] > close.iloc[i+1]
        is_l = close.iloc[i] < close.iloc[i-1] and close.iloc[i] < close.iloc[i+1]
        if (is_h or is_l) and abs(close.iloc[i] - last_p) > atr_val * 1.2:
            pivots.append((df.index[i], close.iloc[i], "H" if is_h else "L"))
            last_p, last_idx = close.iloc[i], i

    clean = []
    for p in pivots:
        if not clean: clean.append(p)
        else:
            last = clean[-1]
            if p[2] == last[2]:
                if (p[2] == "H" and p[1] > last[1]) or (p[2] == "L" and p[1] < last[1]): clean[-1] = p
            else: clean.append(p)

    structure = []
    for i in range(1, len(clean)):
        prev, curr = clean[i-1], clean[i]
        label = ("HH" if curr[1] > prev[1] else "LH") if curr[2] == "H" else ("HL" if curr[1] > prev[1] else "LL")
        structure.append((curr[0], curr[1], label))

    # Calculate Flags using 5-Day Rolling VWAP
    vwap_now = float(vwap.ffill().iloc[-1])
    cci_now = float(cci.iloc[-1])
    pivot_now = structure[-1][2] if structure else "N/A"
    flip_now = "UP" if price_now > vwap_now else "DOWN"
    
    if flip_now == "UP" and cci_now > 0 and pivot_now in ["HH", "HL"]:
        sig_now, col_now = "ALL UP", "#00ff00"
        scenario = "TREND UP / ABOVE 5D VWAP / CCI POSITIVE / HH-HL"
    elif flip_now == "DOWN" and cci_now < 0 and pivot_now == "LL":
        sig_now, col_now = "ALL DOWN", "#ff4b4b"
        scenario = "TREND DOWN / BELOW 5D VWAP / CCI NEGATIVE / LL"
    elif flip_now == "DOWN" and cci_now < 0 and pivot_now in ["HH", "HL"]:
        sig_now, col_now = "MONITOR PIVOT", "#FFA500"
        scenario = f"TREND DOWN / BELOW 5D VWAP / CCI NEGATIVE / HH-HL"
    elif flip_now == "DOWN" and cci_now > 0 and pivot_now == "LL":
        sig_now, col_now = "MONITOR CCI", "#FFA500"
        scenario = f"TREND DOWN / BELOW 5D VWAP / CCI POSITIVE / LL "
    else:
        sig_now, col_now = "NO DIRECTION", "#888"
        scenario = f"NO DIRECTION"    

    vwap_col = "#00ff00" if price_now > vwap_now else "#ff4b4b"
    cci_col = "#00ff00" if cci_now >= 0 else "#ff4b4b"

    return {
        "df": df, "cci": cci, "vwap": vwap, "structure": structure,
        "sig": sig_now, "col": col_now, "pivot": pivot_now, 
        "vwap_val": vwap_now, "price": price_now, "cci_val": cci_now, 
        "flip": flip_now, "scenario": scenario,
        "vwap_col": vwap_col, "cci_col": cci_col
    }

def render_chart(data):
    if not data:
        st.error(f"No data found for {ticker}")
        return

    df, cci, vwap, structure = data["df"], data["cci"], data["vwap"], data["structure"]
    close = df["Close"].squeeze()
    
    # Header Layout
    c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 3, 1.0, 1.0, 0.8, 0.8, 0.8]) 
    c1.markdown(f"<p class='flag-label'>Signal</p><p class='flag-value' style='color:{data['col']}'>{data['sig']}</p>", unsafe_allow_html=True)
    c2.markdown(f"<p class='flag-label'>Scenario</p><p class='flag-value'>{data['scenario']}</p>", unsafe_allow_html=True)    
    c3.markdown(f"<p class='flag-label'>5D VWAP</p><p class='flag-value' style='color:{data['vwap_col']}'>{data['vwap_val']:.2f}</p>", unsafe_allow_html=True)
    c4.markdown(f"<p class='flag-label'>Price Now</p><p class='flag-value'>{data['price']:.2f}</p>", unsafe_allow_html=True)
    c5.markdown(f"<p class='flag-label'>CCI</p><p class='flag-value' style='color:{data['cci_col']}'>{data['cci_val']:.2f}</p>", unsafe_allow_html=True)
    c6.markdown(f"<p class='flag-label'>Flip</p><p class='flag-value'>{data['flip']}</p>", unsafe_allow_html=True)
    c7.markdown(f"<p class='flag-label'>Pivot</p><p class='flag-value'>{data['pivot']}</p>", unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.03)

    # Main Price Trace
    fig.add_trace(go.Scatter(x=df.index, y=close, name="Price", line=dict(color="#4682B4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=vwap, name="5D Rolling VWAP", line=dict(color="orange", width=1.5)), row=1, col=1)

    # CCI Markers on Price
    ob_mask, os_mask = cci > 100, cci < -100
    fig.add_trace(go.Scatter(x=df.index[ob_mask], y=close[ob_mask], mode="markers", name="CCI > 50", marker=dict(color="#ff4b4b", size=6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[os_mask], y=close[os_mask], mode="markers", name="CCI < -50", marker=dict(color="#00ff00", size=6)), row=1, col=1)

    # STRUCTURE LINES (DOT)
    for i in range(1, len(structure)):
        fig.add_trace(go.Scatter(
            x=[structure[i-1][0], structure[i][0]],
            y=[structure[i-1][1], structure[i][1]],
            mode="lines",
            line=dict(dash="dot"),
            showlegend=False
        ), row=1, col=1)

    # LABELS
    for x, y, label in structure:
        color = "green" if "H" in label else "red"
        fig.add_annotation(
            x=x, y=y, text=label, showarrow=True, bgcolor=color,
            font=dict(color="white"), ay=-25 if "H" in label else 25,
            row=1, col=1
        )

    # Invisible Hover Trace
    h_x, h_y, h_t = [s[0] for s in structure], [s[1] for s in structure], [s[2] for s in structure]
    fig.add_trace(go.Scatter(x=h_x, y=h_y, mode="markers", marker=dict(size=10, color="rgba(0,0,0,0)"), 
                             showlegend=False, hovertemplate="<b>%{text}</b> Price: %{y:.2f}<extra></extra>", text=h_t), row=1, col=1)

    # CCI Indicators
    fig.add_trace(go.Scatter(x=df.index, y=cci, name="CCI", line=dict(color="cyan", width=1)), row=2, col=1)
    fig.add_hline(y=100, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-100, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0, line_color="white", line_dash="dot", row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark", height=650, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    st.plotly_chart(fig, use_container_width=True, config={
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']
    })

# ============= TABS =============
tabs = st.tabs(["📅 1-MONTH", "📅 3-MONTH", "📅 6-MONTH"])
with tabs[0]: render_chart(get_and_analyze(ticker, "1mo"))
with tabs[1]: render_chart(get_and_analyze(ticker, "3mo"))
with tabs[2]: render_chart(get_and_analyze(ticker, "6mo"))
