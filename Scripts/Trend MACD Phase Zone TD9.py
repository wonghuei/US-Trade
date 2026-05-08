import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
from datetime import datetime
from zoneinfo import ZoneInfo

# ============================================
# PAGE CONFIG & COMPACT CSS
# ============================================
NY_TZ = pytz.timezone("America/New_York")

st.set_page_config(layout="wide", page_title="MACD-TD9")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
        [data-testid="stMetricValue"] { font-size: 15px !important; font-weight: 700; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 13px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>MACD Phase Box + Volume Profile (TD9)</h1>", unsafe_allow_html=True)

# =====================================================
# TD SEQUENTIAL LOGIC 
# =====================================================
def process_td_sequential_logic(df_input):
    close = df_input["Close"].values
    high = df_input["High"].values
    low = df_input["Low"].values
    n = len(df_input)

    def bars_last_count(cond):
        count = [0] * n
        for i in range(n):
            if cond[i]:
                count[i] = count[i - 1] + 1 if i > 0 else 1
            else:
                count[i] = 0
        return count

    buy_cond  = [close[i] < close[i - 4] if i >= 4 else False for i in range(n)]
    sell_cond = [close[i] > close[i - 4] if i >= 4 else False for i in range(n)]

    buy_count = bars_last_count(buy_cond)
    sell_count = bars_last_count(sell_cond)

    buy_seq = [0] * n
    sell_seq = [0] * n

    last_buy_idx = None
    for i in range(n-1, -1, -1):
        if buy_count[i] > 0:
            last_buy_idx = i
            break

    # latest active sequence
    if last_buy_idx is not None:
        val = buy_count[last_buy_idx]
        for j in range(last_buy_idx, max(-1, last_buy_idx - val), -1):
            if 1 <= buy_count[j] <= 9:
                buy_seq[j] = buy_count[j]

    # completed sequences
    for i in range(n):
        if buy_count[i] == 9:
            for j in range(i, i-9, -1):
                if j >= 0:
                    buy_seq[j] = buy_count[j]

    # =========================
    # SELL SIDE
    # =========================
    last_sell_idx = None
    for i in range(n-1, -1, -1):
        if sell_count[i] > 0:
            last_sell_idx = i
            break

    if last_sell_idx is not None:
        val = sell_count[last_sell_idx]
        for j in range(last_sell_idx, max(-1, last_sell_idx - val), -1):
            if 1 <= sell_count[j] <= 9:
                sell_seq[j] = sell_count[j]

    for i in range(n):
        if sell_count[i] == 9:
            for j in range(i, i-9, -1):
                if j >= 0:
                    sell_seq[j] = sell_count[j]    

    return buy_seq, sell_seq

# =====================================================
# OTHER INDICATORS
# =====================================================
def resample_ohlc(df, rule):
    return df.resample(rule).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()

# ============================================
# FUNCTIONS
# ============================================
def kama_dynamic(price, lookback=20):
    if not isinstance(price, pd.Series):
        price = pd.Series(price.squeeze())
    returns = price.pct_change().dropna()
    volatility_std = returns.rolling(lookback).std().mean() if not returns.empty else 0
    trend_strength = (
        (price.iloc[-lookback:].max() - price.iloc[-lookback:].min())
        / price.iloc[-lookback:].mean()
        if len(price) >= lookback else 0
    )
    n = 5 if volatility_std > 0.02 else 10 if volatility_std > 0.01 else 20
    fast, slow = (2, 20) if trend_strength > 0.05 else (5, 50)
    change = abs(price - price.shift(n))
    vol_sum = abs(price.diff()).rolling(window=n).sum()
    er = (change / vol_sum).fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = pd.Series(index=price.index, dtype=float)
    kama.iloc[:n] = price.iloc[:n].rolling(window=n, min_periods=1).mean()
    for i in range(n, len(price)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (price.iloc[i] - kama.iloc[i-1])
    return kama.ffill()

def kama_angle(df, col="KAMA"):
    kama = df[col]
    close = df["Close"]
    lookbacks = [5, 10, 20, 50, 120, 200]

    for n in lookbacks:
        # Global Standard: Price-based momentum (Script 2)
        df[f"Momentum_{n}D"] = close.pct_change(n) * 100
        
        # Global Standard: Normalized Slope (Script 2)
        # Change in KAMA divided by the number of bars
        slope = (kama - kama.shift(n)) / n
        df[f"Angle_{n}D"] = np.degrees(np.arctan(slope))

        # Efficiency Standard: Vectorized Rank (Script 1)
        df[f"Angle_{n}D_PCTL"] = df[f"Angle_{n}D"].rolling(100).rank(pct=True)

    df["Price_vs_KAMA_%"] = (close - kama) / kama * 100
    return df

    # --- MOMENTUM ---
    df["Momentum_5D"] = kama.pct_change(5) * 100
    df["Momentum_10D"] = kama.pct_change(10) * 100
    df["Momentum_20D"] = kama.pct_change(20) * 100
    df["Momentum_50D"] = kama.pct_change(50) * 100
    df["Momentum_12D"] = kama.pct_change(120) * 100
    df["Momentum_200D"] = kama.pct_change(200) * 100

    # --- PRICE POSITION ---
    df["Price_vs_KAMA_%"] = (df["Close"] - kama) / kama * 100

    # --- PERCENTILE STRENGTH ---
    for col_name in ["Angle_5D", "Angle_10D", "Angle_20D", "Angle_50D", "Angle_120D", "Angle_200D"]:
        df[f"{col_name}_PCTL"] = df[col_name].rolling(100).rank(pct=True)

    return df

# ============================================
# KAMA SIGNAL ENGINE
# ============================================

def tf_state(pctl):
    if pctl > 0.8:
        return "🔥 Strong Trend"
    elif pctl > 0.6:
        return "📈 Trending"
    elif pctl < 0.2:
        return "❄️ Weak"
    else:
        return "⚖️ Neutral"


def kama_trend_display(row):
    states = []

    for tf in ["5D", "10D", "20D", "50D", "120D", "200D"]:
        angle = row[f"Angle_{tf}"]
        if angle > 0:
            states.append(f"{tf}↑")
        else:
            states.append(f"{tf}↓")

    return " | ".join(states)


def final_kama_signal(row):
    up_count = 0
    down_count = 0

    for tf in ["5D", "10D", "20D", "50D", "120D", "200D"]:
        if row[f"Angle_{tf}"] > 0:
            up_count += 1
        else:
            down_count += 1

    price_pos = row["Price_vs_KAMA_%"]

    if up_count == 4:
        if price_pos > 10:
            return "⚠️ OVEREXTENDED (TAKE PROFIT)"
        return "📈 ALL TIMEFRAME UP (HOLD / ADD)"

    elif up_count >= 2:
        return "🟢 SHORT–MID TREND UP (ACCUMULATE)"

    elif down_count == 4:
        if price_pos < -10:
            return "🔻 ALL TIMEFRAME DOWN (WEAK)"
        return "📉 ALL TIMEFRAME DOWN (EXIT / SHORT)"

    return "⚪ NO STRUCTURE (WAIT)"

def get_market_stats_from_df(df_d):
    df = df_d.copy().tail(40) 
    df['DR'] = df['High'] - df['Low']
    df['TR'] = pd.concat([
        df['DR'],
        (df['High']-df['Close'].shift(1)).abs(),
        (df['Low']-df['Close'].shift(1)).abs()
    ], axis=1).max(axis=1)

    atr5, atr10, atr20 = df['TR'].tail(5).mean(), df['TR'].tail(10).mean(), df['TR'].tail(20).mean()
    adr5, adr10, adr20 = df['DR'].tail(5).mean(), df['DR'].tail(10).mean(), df['DR'].tail(20).mean()
    vol5, vol20 = df['Volume'].tail(5).mean(), df['Volume'].tail(20).mean()

    safe_adr = min(adr5, adr10)
    safe_atr = min(atr5, atr10)

    atr_status = "Expanding" if atr5 > atr10 > atr20 else ("Exhausting" if atr5 < atr10 else "Normal")
    adr_status = "Increasing" if safe_adr > adr20 * 1.05 else ("Reduced (Coiling)" if safe_adr < adr20 * 0.95 else "Stable")
    vol_status = "Surge (High Volume)" if vol5 > vol20 else "Quiet (Low Volume)"

    final_f = "Wait"
    if atr_status == "Expanding" and adr_status == "Increasing" and vol_status == "Surge (High Volume)":
        final_f = "Prime Buy"
    elif (adr5 / adr10 - 1) < -0.10 and vol_status == "Surge (High Volume)":
        final_f = "Watch (Tight)"
    elif atr_status == "Expanding":
        final_f = "Watch (Energy)"

    # --- THE CRITICAL FIX: You must RETURN the dictionary ---
    return {
        "atr_vals": f"{atr5:.2f}/{atr10:.2f}/{atr20:.2f}",
        "adr_vals": f"{adr5:.2f}/{adr10:.2f}/{adr20:.2f}",
        "adr_min": f"{safe_adr:.2f}",
        "atr_min": f"{safe_atr:.2f}",
        "final_f": final_f,
        "atr_desc": atr_status,
        "adr_desc": adr_status,
        "vol_desc": vol_status
    }

def calculate_daily_vwap(df):
    df = df.copy()

    # Typical price
    df["TP"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # Group by date (VERY IMPORTANT)
    df["Date"] = df.index.date

    df["Cum_TPVol"] = (df["TP"] * df["Volume"]).groupby(df["Date"]).cumsum()
    df["Cum_Vol"] = df["Volume"].groupby(df["Date"]).cumsum()

    df["VWAP"] = df["Cum_TPVol"] / df["Cum_Vol"]

    return df["VWAP"]

def sma_moomoo(series, n, m):
    return series.ewm(alpha=m/n, adjust=False).mean()

def calculate_macd_kdj_logic(df, n_period=20):
    c, h, l = df['Close'], df['High'], df['Low']
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_raw = (dif - dea) * 2
    m_max = pd.concat([dif, dea, macd_raw], axis=1).rolling(n_period).max().max(axis=1)
    m_min = pd.concat([dif, dea, macd_raw], axis=1).rolling(n_period).min().min(axis=1)
    rsv = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-9) * 100
    K = sma_moomoo(rsv, 3, 1)
    D = sma_moomoo(K, 3, 1)
    J = 3*K - 2*D
    k_max = pd.concat([K, D, J], axis=1).rolling(n_period).max().max(axis=1)
    k_min = pd.concat([K, D, J], axis=1).rolling(n_period).min().min(axis=1)
    m1, a1 = (m_max + m_min) / 2, (m_max - m_min).replace(0, np.nan)
    m2, a2 = (k_max + k_min) / 2, (k_max - k_min)
    base = (0 - m1) * a2 / a1 + m2
    df['J_NEW'] = J - base
    df['MACD_KDJ'] = (macd_raw - m1) * a2 / a1 + m2 - base
    return df

# ============================================
# SAFETY WRAPPED DATA FETCHING (UNIFIED)
# ============================================
NY_TZ = ZoneInfo("America/New_York")

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)
    
col_input, col_btn, _ = st.columns([2, 2, 2])

with col_input:
    st.markdown(
    f"<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>"
    f"Enter Stock Ticker  |  Last refresh (AMS / NY) : "
    f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
    )
    ticker = st.text_input("Enter Ticker:", value="SOXX", autocomplete="off", label_visibility="collapsed").strip()
    
with col_btn:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    run_scan = st.button("🔄")

# ============================================
# DOWNLOAD
# ============================================
@st.cache_data
def load_all_mtf(ticker):
    d = yf.download(ticker, period="1y", interval="1d", progress=False)
    w = yf.download(ticker, period="3y", interval="1wk", progress=False)
    m = yf.download(ticker, period="5y", interval="1mo", progress=False)
    
    # Strictly download Pre/Post and localize to New York
    now = yf.download(ticker, period="1d", interval="5m", prepost=True, progress=False)

    for df in [d, w, m, now]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    
    # NY Timezone Safety Wrap
    if not now.empty:
        if now.index.tz is None:
            now.index = now.index.tz_localize('UTC').tz_convert(NY_TZ)
        else:
            now.index = now.index.tz_convert(NY_TZ)

    return d.dropna(), w.dropna(), m.dropna(), now.dropna()

df_d, df_w, df_m, df_now = load_all_mtf(ticker)
# --- KAMA + ANGLE PRE-CALC FOR ALL TABS ---
for df in [df_d, df_w, df_m]:
    df["KAMA"] = kama_dynamic(df["Close"])
    df = kama_angle(df)

if df_d.empty:
    st.error("No data found.")
    st.stop()

# Use localized Price Zone NOW logic
curr_p = float(df_now['Close'].iloc[-1]) if not df_now.empty else float(df_d['Close'].iloc[-1])
last_time_ny = df_now.index[-1].strftime('%H:%M:%S') if not df_now.empty else "Market Closed"

# Standard indicators
h52, l52 = float(df_d["High"].tail(252).max()), float(df_d["Low"].tail(252).min())
h120, l120 = float(df_d["High"].tail(120).max()), float(df_d["Low"].tail(120).min())
h30, l30 = float(df_d["High"].tail(30).max()), float(df_d["Low"].tail(30).min())
h60, l60 = float(df_d["High"].tail(60).max()), float(df_d["Low"].tail(60).min())
h90, l90 = float(df_d["High"].tail(90).max()), float(df_d["Low"].tail(90).min())
m_stats = get_market_stats_from_df(df_d)

def render_chart(df_input, interval_label):
    df = df_input.copy()

    # =========================
    # GAP (缺口) DETECTION
    # =========================
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)

    df['gap_up'] = df['Low'] > df['prev_high']
    df['gap_down'] = df['High'] < df['prev_low']

    # =========================
    # ADD THIS BLOCK HERE 👇
    # =========================
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    N = 14
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    macd_raw = (dif - dea) * 2
    m_max = pd.concat([dif, dea, macd_raw], axis=1).rolling(N).max().max(axis=1)
    m_min = pd.concat([dif, dea, macd_raw], axis=1).rolling(N).min().min(axis=1)
    rsv = (c - l.rolling(14).min()) / (h.rolling(14).max() - l.rolling(14).min() + 1e-9) * 100
    K = sma_moomoo(rsv, 3, 1)
    D = sma_moomoo(K, 3, 1)
    J = 3*K - 2*D
    k_max = pd.concat([K, D, J], axis=1).rolling(N).max().max(axis=1)
    k_min = pd.concat([K, D, J], axis=1).rolling(N).min().min(axis=1)
    m1, a1 = (m_max + m_min) / 2, (m_max - m_min).replace(0, np.nan)
    m2, a2 = (k_max + k_min) / 2, (k_max - k_min)
    base = (0 - m1) * a2 / a1 + m2
    df["J_NEW"] = J - base
    df["MACD_KDJ"] = (macd_raw - m1) * a2 / a1 + m2 - base

    # =========================
    # TD SEQUENTIAL INJECTION
    # =========================
    b9, s9 = process_td_sequential_logic(df)
    df['TD_B9'] = b9
    df['TD_S9'] = s9
    
    vol_sum = float(df["Volume"].sum())
    
    if len(df) > 1 and vol_sum > 0:
        counts, bins = np.histogram(df['Close'], bins=50, weights=df['Volume'])
        poc_idx = int(np.argmax(counts))
        poc_price = (bins[poc_idx] + bins[poc_idx+1]) / 2
        bin_centers = (bins[:-1] + bins[1:]) / 2
    else:
        counts, bin_centers = np.array([0]), np.array([curr_p])
        poc_price = curr_p

    
    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = (df["EMA12"] - df["EMA26"] - (df["EMA12"] - df["EMA26"]).ewm(span=9).mean()) * 2
    df["MA20"] = df["Close"].rolling(window=20).mean()

    # Get intraday data for proper VWAP
    intraday = yf.download(ticker, period="1d", interval="15m", prepost=True, progress=False)

    df_intraday = yf.download(ticker, period="1d", interval="5m")
    tp = (df_intraday['High'] + df_intraday['Low'] + df_intraday['Close']) / 3
    df_intraday['VWAP'] = (tp * df_intraday['Volume']).cumsum() / df_intraday['Volume'].cumsum()
    vwap = df_intraday['VWAP'].iloc[-1]

    # Use the result from the function
    m_stats = get_market_stats_from_df(df_input)

    r1 = st.columns(5, gap="small")
    r1[0].metric("30D H / L", f"{h30:.2f} / {l30:.2f}")
    r1[1].metric("60D H / L", f"{h60:.2f} / {l60:.2f}")
    r1[2].metric("90D H / L", f"{h90:.2f} / {l90:.2f}")
    r1[3].metric("120D H / L", f"{h120:.2f} / {l120:.2f}")
    r1[4].metric("52W H / L", f"{h52:.2f} / {l52:.2f}")

    # --- DIAGNOSTIC INFO BOX ---
    st.info(f"🔍 **Logic Diagnostic:** Volatility is **{m_stats['atr_desc']}** | Range is **{m_stats['adr_desc']}** | Volume is **{m_stats['vol_desc']}**")

    # ============================================
    # INSERT NEW TREND MEMORY LOGIC HERE
    # ============================================
    # --- 1. RE-ESTABLISH THE ACTIVE BOX EXTREMES ---
    active_macd_phase = "bull" if df["MACD"].iloc[-1] > 0 else "bear"
    
    # --- 1. IDENTIFY THE START OF THE CURRENT MACD PHASE ---
    macd_signals = (df["MACD"] > 0).astype(int)
    phase_changes = macd_signals.diff().fillna(0) != 0
    # Find the most recent flip in MACD (start of the current box)
    change_indices = df.index[phase_changes].tolist()
    last_phase_start_idx = change_indices[-1] if change_indices else df.index[0]
    
    # --- 2. CALCULATE EXTREMES: BEFORE vs AFTER ---
    # 'box_after' is everything in the current trend including the latest candle
    box_after = df.loc[last_phase_start_idx:]
    
    # 'box_before' is everything in the current trend EXCEPT the latest candle
    if len(box_after) > 1:
        box_before = box_after.iloc[:-1]
        h_before = float(box_before["High"].max())
        l_before = float(box_before["Low"].min())
    else:
        # If this is the FIRST candle of a new MACD phase, 'before' is just the current open
        h_before = float(df["Open"].iloc[-1])
        l_before = float(df["Open"].iloc[-1])

    # Final "After" values (Current State)
    h_after = float(box_after["High"].max())
    l_after = float(box_after["Low"].min())
    
    # Midpoints
    mid_before = (h_before + l_before) / 2
    mid_after = (h_after + l_after) / 2
    
    # Shifts
    shift_h = h_after - h_before
    shift_l = l_after - l_before
    shift_mid = mid_after - mid_before

    # --- 3. RENDER THE BOX SHIFT ANALYSIS TABLE ---
    st.markdown(f"**Box Shift Analysis** (Current MACD Phase)")
    
    shift_df = pd.DataFrame({
        "Level Type": ["Active High", "Active Low", "Active Midpoint"],
        "Price Active": [f"{h_before:.2f}", f"{l_before:.2f}", f"{mid_before:.2f}"],
        "Price Last": [f"{h_after:.2f}", f"{l_after:.2f}", f"{mid_after:.2f}"],
        "Shift Amt": [f"{shift_h:+.2f}", f"{shift_l:+.2f}", f"{shift_mid:+.2f}"],
        "Status": [
            "OUTBREAK (HH)" if shift_h > 0.001 else "Holding",
            "OUTBREAK (LL)" if shift_l < -0.001 else "Holding",
            "Relocating" if abs(shift_mid) > 0.001 else "Stable"
        ]
    })
    
    st.dataframe(shift_df, hide_index=True, width="stretch")

    # Volume Profile Plot
    fig = make_subplots(
        rows=4, cols=2, 
        shared_xaxes=True, 
        shared_yaxes=True,
        column_widths=[0.9, 0.1], 
        row_heights=[0.55, 0.20, 0.20, 0.05], 
        vertical_spacing=0.02, 
        horizontal_spacing=0.01,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}], 
            [{"secondary_y": False}, None], 
            [{"secondary_y": False}, None], 
            [{"secondary_y": False}, None]
        ]
    )

    # Price Chart
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price", increasing_line_width=0.8, decreasing_line_width=0.8), row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            mode="lines",
            line=dict(width=1, dash="dot", color="pink"),
            name="MA20"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["EMA26"],
            mode="lines",
            line=dict(width=1, color="cyan"),
            name="EMA26"
        ),
        row=1,
        col=1
    )

    # =========================
    # GAP VISUALIZATION (缺口区间)
    # =========================
    for i, row in enumerate(df.itertuples()):    
        # auto width based on candle spacing
        if i < len(df) - 1:
            half_width = (df.index[i+1] - df.index[i]) * 0.4
        else:
            half_width = (df.index[i] - df.index[i-1]) * 0.4

        # =========================
        # GAP UP (向上跳空)
        # =========================
        if row.gap_up:
            fig.add_shape(
                type="rect",
                x0=df.index[i] - half_width,
                x1=df.index[i] + half_width,
                y0=row.prev_high,
                y1=row.Low,
                line=dict(color="white", width=2),
                fillcolor="rgba(255, 215, 0, 0.03)",   # light gold
                layer="below"
            )

        # =========================
        # GAP DOWN (下突缺口)
        # =========================
        if row.gap_down:
            fig.add_shape(
                type="rect",
                x0=df.index[i] - half_width,
                x1=df.index[i] + half_width,
                y0=row.High,
                y1=row.prev_low,
                line=dict(color="magenta", width=2),
                fillcolor="rgba(153, 77, 153, 0.05)",  # light purple
                layer="below"
            )
    
    # =========================
    # TD ANNOTATIONS (PLOTLY)
    # =========================
    for i, row in enumerate(df.itertuples()):
        if row.TD_B9 > 0:
            fig.add_annotation(
                x=row.Index,
                y=row.Low * 0.99,
                text=str(int(row.TD_B9)),
                showarrow=False,
                font=dict(color="yellow", size=11),
                xanchor="center",
                yanchor="top"
            )

        if row.TD_S9 > 0:
            fig.add_annotation(
                x=row.Index,
                y=row.High * 1.01,
                text=str(int(row.TD_S9)),
                showarrow=False,
                font=dict(color="skyblue", size=11),
                xanchor="center",
                yanchor="bottom"
            )
            
    # 1. Volume Profile (Horizontal Bars)
    fig.add_trace(go.Bar(
        x=counts, y=bin_centers, orientation='h',
        marker=dict(color='rgba(255, 223, 128, 0.5)', line=dict(color='rgba(255, 223, 128, 0.5)', width=0.5)),
        name="Volume Profile"
    ), row=1, col=2)

    # 2. POC Line
    fig.add_shape(type="line", x0=0, x1=max(counts), y0=poc_price, y1=poc_price,
                  line=dict(color="red", width=1, dash="dash"), row=1, col=2)
    fig.add_annotation(x=max(counts), y=poc_price, text="POC", showarrow=False, xanchor="left", font=dict(color="red", size=12), row=1, col=2)

    # 3. NOW Line
    fig.add_shape(type="line", x0=0, x1=max(counts), y0=curr_p, y1=curr_p,
                  line=dict(color="#00FF00", width=1), row=1, col=2)
    fig.add_annotation(x=max(counts), y=curr_p, text="PRICE NOW", showarrow=False, xanchor="left", font=dict(color="#00FF00", size=12), row=1, col=2)

    # 4. VWAP Line
    fig.add_shape(type="line", x0=0, x1=max(counts), y0=vwap, y1=vwap,
                  line=dict(color="white", width=1, dash="dash"), row=1, col=2)
    fig.add_annotation(x=max(counts), y=vwap, text="VWAP", showarrow=False, xanchor="left", font=dict(color="white", size=12), row=1, col=2)

    # 5. 90D/120D/52W Levels Loop
    level_list = [(h30, "30DH", "yellow"), (l30, "30DL", "yellow"), (h60, "60DH", "orange"), (l60, "60DL", "orange"), (h90, "90DH", "cyan"), (l90, "90DL", "cyan"), (h120, "120DH", "pink"), (l120, "120DL", "pink")]
    if interval_label in ["1wk", "1mo"]:
        level_list.extend([(l52, "52WL", "gray"), (h52, "52WH", "gray")])

    for val, label, col in level_list:
        fig.add_shape(type="line", x0=0, x1=max(counts), y0=val, y1=val,
                      line=dict(color=col, width=1, dash="dashdot"), row=1, col=2)
        fig.add_annotation(x=max(counts), y=val, text=label, showarrow=False, xanchor="left", font=dict(color=col, size=12), row=1, col=2)

    # Volume
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=poc_price, y1=poc_price, line=dict(color="red", width=2, dash="dash"), row=1, col=1)
    vol_c = ['#26a69a' if r.Close >= r.Open else '#ef5350' for r in df.itertuples()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=vol_c, name="Volume"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['J_NEW'], line=dict(color='pink', width=1), name="KDJ-14D"), row=3, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_KDJ'], marker_color=['green' if v >= 0 else 'red' for v in df['MACD_KDJ']], name="MACD-14D"), row=3, col=1)
    
    # MACD Phase Logic
    phase, start_idx, h_val, l_val = None, None, None, None
    for i in range(len(df)):
        m_val, hi, lo = float(df["MACD"].iloc[i]), float(df["High"].iloc[i]), float(df["Low"].iloc[i])
        if phase is None: phase, start_idx, h_val, l_val = ("bull" if m_val > 0 else "bear"), i, hi, lo
        else:
            curr_phase = "bull" if m_val > 0 else "bear"
            if curr_phase == phase: h_val, l_val = max(h_val, hi), min(l_val, lo)
            else:
                c, lc = ("rgba(0,255,255,0.1)", "green") if phase == "bull" else ("rgba(255,255,0,0.1)", "orange")
                x0, x1 = df.index[start_idx], df.index[i-1]
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=l_val, y1=h_val, fillcolor=c, line=dict(width=0), layer="below", row=1, col=1)
                fig.add_trace(go.Scatter(x=[x0, x1], y=[h_val, h_val], mode="lines", line=dict(color=lc, width=1), showlegend=False), row=1, col=1)
                fig.add_trace(go.Scatter(x=[x0, x1], y=[l_val, l_val], mode="lines", line=dict(color=lc, width=1), showlegend=False), row=1, col=1)
                phase, start_idx, h_val, l_val = curr_phase, i, hi, lo

    if phase is not None:
        line_c, x0, x_f = ("cyan" if phase == "bull" else "orange"), df.index[start_idx], df.index[-1] + pd.Timedelta(days=5)
        p_name, mid_val = ("Bull" if phase == "bull" else "Bear"), (h_val + l_val) / 2
        fig.add_shape(type="rect", x0=x0, x1=x_f, y0=l_val, y1=h_val, fillcolor="rgba(128,128,128,0.1)", line=dict(width=0), layer="below", row=1, col=1)
        fig.add_trace(go.Scatter(x=[x0, x_f], y=[h_val, h_val], mode="lines", line=dict(color=line_c, width=1, dash="dashdot"), showlegend=False, name=f"Active {p_name} TOP"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[x0, x_f], y=[mid_val, mid_val], mode="lines", line=dict(color="pink", width=1, dash="dashdot"), showlegend=False, name=f"Active {p_name} 50%"), row=1, col=1)
        fig.add_annotation(x=x_f, y=mid_val, text=f"{mid_val:.2f}", showarrow=False, yanchor="bottom", yshift=9, font=dict(color="pink", size=13), row=1, col=1)
        fig.add_trace(go.Scatter(x=[x0, x_f], y=[l_val, l_val], mode="lines", line=dict(color=line_c, width=1, dash="dashdot"), showlegend=False, name=f"Active {p_name} BOT"), row=1, col=1)
        
    fig.update_layout(
        height=900, 
        template="plotly_dark", 
        xaxis_rangeslider_visible=False,
        dragmode='drawline', 
        # --- Drawing Tool Color Settings ---
        newshape=dict(
            line=dict(color="white", width=2),
            fillcolor="rgba(255, 255, 255, 0.2)"
        ),
        # --- Legend moved to Left, Single Row ---
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="left", 
            x=0,
            bgcolor="rgba(0,0,0,0)"
        ),
        # --- Increased Top Margin to prevent overlap ---
        margin=dict(t=80, b=20, l=20, r=60)
    )

    st.plotly_chart(
        fig, 
        width="stretch", 
        config={
            'modeBarButtonsToAdd': [
                'drawline',
                'drawopenpath',
                'drawclosedpath',
                'drawcircle',
                'drawrect',
                'eraseshape'
            ]
        }
    )

def create_kama_plotly(df, ticker):
    plot_df = df.tail(120).copy()
    x = plot_df.index

    fig = go.Figure()

    # --- Angle Lines ---
    fig.add_trace(go.Scatter(
        x=x, y=plot_df["Angle_5D"],
        mode="lines",
        name="5D Angle",
        line=dict(color="#1f77b4", width=2, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=x, y=plot_df["Angle_10D"],
        mode="lines",
        name="10D Angle",
        line=dict(color="purple", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=plot_df["Angle_20D"],
        mode="lines",
        name="20D Angle",
        line=dict(color="green", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=plot_df["Angle_50D"],
        mode="lines",
        name="50D Angle",
        line=dict(color="#d62728", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=plot_df["Angle_120D"],
        mode="lines",
        name="120D Angle",
        line=dict(color="yellow", width=1)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=plot_df["Angle_200D"],
        mode="lines",
        name="200D Angle",
        line=dict(color="orange", width=1)
    ))

    # --- Zero Line ---
    fig.add_hline(y=0, line_width=2, line_dash="dot", line_color="gray")

    # --- Momentum Bars (secondary axis feel) ---
    colors = ['lightgreen' if v >= 0 else 'pink' for v in plot_df['Momentum_20D']]

    fig.add_trace(go.Bar(
        x=x,
        y=plot_df["Momentum_20D"],
        name="20D Momentum %",
        marker_color=colors,
    ))

    # --- Layout ---
    st.markdown("<p style='margin-bottom: 0px; color: steelblue; font-size: 20px; font-weight: bold;'>KAMA Multi-Angle & Momentum</p>", unsafe_allow_html=True)
    fig.update_layout(        
        template="plotly_dark",
        height=600,
        hovermode="x unified",
        xaxis=dict(title=""),
        #yaxis=dict(title="KAMA Angle (°)"),
        barmode="overlay",
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60, l=20, r=20, b=20)
    )

    return fig

# ============================================
# NARROW 20D SHIFT HISTORY (SORTABLE & FULL DATE)
# ============================================
def render_shift_history(df_input):
    df = df_input.tail(60).copy() 
    df["EMA12"] = df["Close"].ewm(span=12).mean()
    df["EMA26"] = df["Close"].ewm(span=26).mean()
    df["MACD"] = (df["EMA12"] - df["EMA26"] - (df["EMA12"] - df["EMA26"]).ewm(span=9).mean()) * 2
    
    history_rows = []
    
    for i in range(len(df)-20, len(df)):
        current_date = df.index[i]
        temp_df = df.iloc[:i+1]
        
        # MACD Phase Logic
        macd_signals = (temp_df["MACD"] > 0).astype(int)
        phase_changes = macd_signals.diff().fillna(0) != 0
        change_indices = temp_df.index[phase_changes].tolist()
        phase_start_idx = change_indices[-1] if change_indices else temp_df.index[0]
        
        # Current Box
        box_data = temp_df.loc[phase_start_idx:]
        h_now, l_now = float(box_data["High"].max()), float(box_data["Low"].min())
        mid_now = (h_now + l_now) / 2
        
        # Previous Day for Shift
        if len(temp_df) > 1:
            prev_slice = temp_df.iloc[:-1].loc[phase_start_idx:]
            if not prev_slice.empty:
                h_prev, l_prev = float(prev_slice["High"].max()), float(prev_slice["Low"].min())
            else:
                h_prev, l_prev = float(temp_df["Open"].iloc[i]), float(temp_df["Open"].iloc[i])
        else:
            h_prev, l_prev = h_now, l_now
            
        mid_prev = (h_prev + l_prev) / 2
        s_h, s_l, s_mid = h_now - h_prev, l_now - l_prev, mid_now - mid_prev
        
        # Action Logic
        if s_h > 0 and s_l >= 0: act = "⬆️ UP"
        elif s_l < 0 and s_h <= 0: act = "⬇️ DN"
        elif s_h > 0 and s_l < 0: act = "↔️ EXP"
        else: act = "🔒 COIL"

        history_rows.append({
            "Date": current_date.strftime('%Y-%m-%d'),
            "Ph": "BULL" if temp_df["MACD"].iloc[-1] > 0 else "BEAR",
            "Mid": round(mid_now, 2),
            "Δ Mid": round(s_mid, 2),
            "Δ H": round(s_h, 2),
            "Δ L": round(s_l, 2),
            "Action": act
        })

    hist_df = pd.DataFrame(history_rows).iloc[::-1]
    
    # Render table with column width control
    st.dataframe(
        hist_df, 
        width="stretch", 
        hide_index=True,
        column_config={
            "Date": st.column_config.TextColumn("Date", width="medium"),
            "Ph": st.column_config.TextColumn("Ph", width="small"),
            "Mid": st.column_config.NumberColumn("Mid", format="%.2f", width="small"),
            "Δ Mid": st.column_config.NumberColumn("Δ Mid", format="%+.2f", width="small"),
            "Δ H": st.column_config.NumberColumn("Δ H", format="%+.2f", width="small"),
            "Δ L": st.column_config.NumberColumn("Δ L", format="%+.2f", width="small"),
            "Action": st.column_config.TextColumn("Action", width="small"),
        }
    )

# ============================================
# TABS (UPDATED)
# ============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Daily (Short Term)", 
    "📈 Weekly (Mid Term)", 
    "📊 Monthly (Long Term)",
    "📅 20D Box Shift Analysis",
    "📐 KAMA Angle"
])

with tab1: render_chart(df_d, "1d")
with tab2: render_chart(df_w, "1wk")
with tab3: render_chart(df_m, "1mo")
with tab4: render_shift_history(df_d)
with tab5:
    latest = df_d.iloc[-1]

    signal = final_kama_signal(latest)
    structure = kama_trend_display(latest)

    color_map = {
        "📈 ALL TIMEFRAME UP (HOLD / ADD)": "#00ff00",
        "🟢 SHORT–MID TREND UP (ACCUMULATE)": "#00cc66",
        "⚠️ OVEREXTENDED (TAKE PROFIT)": "#ffcc00",
        "📉 ALL TIMEFRAME DOWN (EXIT / SHORT)": "#ff0000",
        "🔻 ALL TIMEFRAME DOWN (WEAK)": "#ff6666",
        "⚪ NO STRUCTURE (WAIT)": "#999999"
    }

    bg = color_map.get(signal, "#333333")

    st.markdown(
        f"""
        <div style="background-color:{bg}; padding:10px; border-radius:10px; font-weight:500;">
            SIGNAL: {signal} | STRUCTURE: {structure}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- METRICS ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mom 5D / 10D", f"{latest['Momentum_5D']:.2f}% / {latest['Momentum_10D']:.2f}%")
    c2.metric("Mom 10D / 20D", f"{latest['Momentum_10D']:.2f}% / {latest['Momentum_20D']:.2f}%")
    c3.metric("Mom 20D / 50D", f"{latest['Momentum_20D']:.2f}% / {latest['Momentum_50D']:.2f}%")
    c4.metric("KAMA", f"{latest['KAMA']:.2f}")
    c5.metric("Price vs KAMA", f"{latest['Price_vs_KAMA_%']:.2f}%")

    d1, d2, d3, d4, d5 = st.columns(5)
    for col, d in zip(["5D","10D","20D","50D","120D"], [d1,d2,d3,d4,d5]):
        d.metric(f"{col} Angle", f"{latest[f'Angle_{col}']:.2f}°")
        d.metric(f"{col} Strength", f"{latest[f'Angle_{col}_PCTL']:.2f}")
        d.write(tf_state(latest[f'Angle_{col}_PCTL']))

    fig = create_kama_plotly(df_d, ticker)
    st.plotly_chart(fig, width="stretch")
