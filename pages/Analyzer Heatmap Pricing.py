import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# CCS
# =====================================================
st.set_page_config(layout="wide", page_title="Heatmap Pricing")

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
        [data-testid="stMetricValue"] { font-size: 15px !important; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 13px !important; }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
    [data-testid="stDataFrame"] {
        width: 100% !important;
    }
    .main {
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔥 Pricing Heatmap</h1>", unsafe_allow_html=True)

# =====================================================
# REGIME BOX
# =====================================================
def market_regime_box(label, flags):
    return f"""
    <div style="
        background-color: #0e1117;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #333;
        text-align: center;
        line-height: 1.1;
        margin-bottom: 0px;">
        <div style="color: #666; font-size: 14px; font-weight: 700; margin: 0px; padding: 0px;">
            {label}
        </div>
        <div style="color: {flags['vol'][1]}; font-size: 15px; font-weight: 700; margin: 0px; padding: 0px;">
            {flags['vol'][0]}
        </div>
        <div style="color: {flags['flow'][1]}; font-size: 15px; font-weight: 700; margin: 0px; padding: 0px;">
            {flags['flow'][0]}
        </div>
        <div style="color: {flags['struct'][1]}; font-size: 15px; font-weight: 700; margin: 0px; padding-top: 4px; border-top: 1px solid #222;">
            {flags['struct'][0]}
        </div>
    </div>
    """
# =====================================================
# INPUT
# =====================================================
NY_TZ = ZoneInfo("America/New_York")

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

col_input, col_refresh = st.columns([2, 2])

with col_input:
    st.markdown(
    f"<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>"
    f"Enter Stock Ticker  |  Last refresh (AMS / NY) : "
    f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
    )
    ticker = st.text_input("Enter Ticker", value="APP", autocomplete="off", label_visibility="collapsed").strip().upper()

with col_refresh:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("🔄"):
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.cache_data.clear()
        st.rerun()

periods = [
    1, 5, 10, 20, 30, 40, 50, 60,
    70, 80, 90, 100,
    110, 120, 130, 140, 150,
    160, 170, 180, 190, 200
]

poc_periods = [20, 60, 120, 160, 200, 250]

# =====================================================
# DATA
# =====================================================
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    df = df.dropna()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

df = load_data(ticker)

if df.empty or len(df) < 300:
    st.warning("")

# =====================================================
# 🔥 ADD THIS (5M DATA FOR SESSION VWAP)
# =====================================================
# =====================================================
# 🔥 ADD THIS (5M DATA FOR SESSION VWAP)
# =====================================================
df_5m = yf.download(ticker, period="1d", interval="5m", prepost=True, progress=False)
df_5m = df_5m.dropna()

# --- ADD THESE TWO LINES TO FIX THE ERROR ---
if isinstance(df_5m.columns, pd.MultiIndex):
    df_5m.columns = df_5m.columns.get_level_values(0)
# ---------------------------------------------

# =====================================================
# INDICATORS
# =====================================================
def calc_atr(df, p):
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False).mean()

def calc_adr(df, p):
    return (df['High'] - df['Low']).ewm(alpha=1/p, adjust=False).mean()

def calc_kama(price, period=10):
    return price.ewm(span=period, adjust=False).mean()  # simplified stable

def calc_vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    return (tp * df['Volume']).cumsum() / df['Volume'].cumsum()

def calc_min_adr(df, p):
    daily_range = df['High'] - df['Low']
    return daily_range.rolling(p).min()

def calc_last_vwap(df_5m, last_date):
    df_day = df_5m[df_5m.index.date == last_date]

    tp = (df_day['High'] + df_day['Low'] + df_day['Close']) / 3
    return (tp * df_day['Volume']).sum() / df_day['Volume'].sum()

def ema_series(df, span=20):
    return df['Close'].ewm(span=span, adjust=False).mean()

def kama_series(df, er_period=10, fast=2, slow=30):
    price = df['Close']

    change = price.diff(er_period).abs()
    volatility = price.diff().abs().rolling(er_period).sum()

    er = change / volatility
    er = er.fillna(0)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=price.index, dtype=float)
    kama.iloc[0] = price.iloc[0]

    for i in range(1, len(price)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (price.iloc[i] - kama.iloc[i-1])

    return kama

def metric_poc(window):
    tp = ((window['High'] + window['Low'] + window['Close']) / 3).round(2)
    vp = window.groupby(tp)['Volume'].sum()
    return vp.idxmax()

# =====================================================
# flags
# =====================================================
def build_flags(df, start, end):
    window = df.iloc[-end:]

    close = window["Close"]

    # --- ADR / ATR ---
    adr = (window["High"] - window["Low"]).mean()
    atr = calc_atr(window, 14).iloc[-1]

    if adr > atr:
        vol_flag = "Increase Volume"
        vol_color = "green"
    else:
        vol_flag = "Reduce Volume"
        vol_color = "#ff5555"

    # --- KAMA vs EMA ---
    kama = calc_kama(close, 10).iloc[-1]
    ema = close.ewm(span=20).mean().iloc[-1]

    if kama > ema:
        flow_flag = "Trend up"
        flow_color = "green"
    else:
        flow_flag = "Chopping"
        flow_color = "#ff5555"

    # --- HH / LL structure ---
    hh = window["High"].max()
    ll = window["Low"].min()
    current_close = close.iloc[-1]
    avg_price = close.mean()

    if current_close > avg_price:
        struct_flag = "Structure Up"
        struct_color = "green"  # Green
    elif current_close < avg_price:
        struct_flag = "Structure Down"
        struct_color = "#ff3333"  # Red
    else:
        struct_flag = "Stable"
        struct_color = "#aaaaaa"  # Grey

    return {
        "vol": (vol_flag, vol_color),
        "flow": (flow_flag, flow_color),
        "struct": (struct_flag, struct_color)
    }

# =====================================================
# METRICS (VALUES ONLY)
# =====================================================
def metric_open(window):
    return window['Open'].iloc[0]

def metric_high(window):
    return window['High'].iloc[0]

def metric_low(window):
    return window['Low'].iloc[0]

def metric_close(window):
    return window['Close'].iloc[0]

def metric_ma(window):
    return window['Close'].mean()

def metric_ema(window):
    return window['Close'].ewm(span=len(window), adjust=False).mean().iloc[-1]

def adr_series(df):
    return df['High'] - df['Low']

def atr_series(df):
    return pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)

def metric_vwap(window):
    tp = (window['High'] + window['Low'] + window['Close']) / 3
    return (tp * window['Volume']).sum() / window['Volume'].sum()

def metric_kama(window):
    price = window['Close']

    if len(price) < 2:
        return price.iloc[-1]

    # Efficiency Ratio
    change = abs(price.iloc[-1] - price.iloc[0])
    volatility = price.diff().abs().sum()

    if volatility == 0:
        return price.iloc[-1]

    er = change / volatility

    # smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # KAMA recursive approximation inside window
    kama = price.iloc[0]

    for i in range(1, len(price)):
        kama = kama + sc * (price.iloc[i] - kama)

    return kama

def metric_hh(window):
    return window['High'].max()

def metric_ll(window):
    return window['Low'].min()

# =====================================================
# 🔥 KPI SECTION (LAST TRADING DAY SNAPSHOT)
# =====================================================
last = df.iloc[-1]
last_date = df.index[-1].date()

# Get the most recent 5-minute bar for live pricing
if not df_5m.empty:
    live_last = df_5m.iloc[-1]
    # Current "Live" Price
    curr_p = float(live_last['Close'])
    # Today's session high/low from 5m data
    today_date = df_5m.index[-1].date()
    df_today = df_5m[df_5m.index.date == today_date]
    
    high_1d = float(df_today['High'].max())
    low_1d = float(df_today['Low'].min())
    open_1d = float(df_today['Open'].iloc[0])
else:
    # Fallback to daily if 5m fails
    curr_p = float(df.iloc[-1]['Close'])
    high_1d = float(df.iloc[-1]['High'])
    low_1d = float(df.iloc[-1]['Low'])
    open_1d = float(df.iloc[-1]['Open'])

# Keep your existing VWAP calculation (already uses 5m)
vwap_1d = float(calc_last_vwap(df_5m, last_date))

# 1. Calculate the Raw True Range (TR) Series
tr = pd.concat([
    df['High'] - df['Low'],
    (df['High'] - df['Close'].shift()).abs(),
    (df['Low'] - df['Close'].shift()).abs()
], axis=1).max(axis=1)

# 2. Update ATR to use Mean (Tail 5/10) instead of iloc Snapshot
atr_1d  = float(tr.iloc[-1])
atr_5d  = float(tr.tail(5).mean())   # Changed from tr.iloc[-5]
atr_10d = float(tr.tail(10).mean())  # Changed from tr.iloc[-10]

# 3. Standardize ADR to use Tail(n).mean() for consistency
daily_range = df['High'] - df['Low']
adr1   = float(daily_range.iloc[-1])
adr5   = float(daily_range.tail(5).mean())
adr10  = float(daily_range.tail(10).mean())

# 4. Standardize Min ADR (Rolling Min of last N days)
minadr5  = float(daily_range.tail(5).min())
minadr10 = float(daily_range.tail(10).min())

# KAMA
kama_1d = float(calc_kama(df['Close'], 10).iloc[-1])

# =====================================================
# 🔥 KPI SECTION (ADD POC HERE)
# =====================================================
# 3 Months = ~63 trading days | 6 Months = ~126 trading days
poc_3m = metric_poc(df.iloc[-60:])
poc_6m = metric_poc(df.iloc[-120:])

# =====================================================
# BUILD MATRIX (VALUES ONLY)
# =====================================================
def build_matrix(df, func):
    result = pd.DataFrame(index=df.index)

    for p in periods:
        values = []

        for i in range(len(df)):
            if i < p - 1:
                values.append(np.nan)
            else:
                window = df.iloc[i-p+1:i+1]  # correct rolling window
                values.append(func(window))   # ONLY ONE ARGUMENT

        result[f"{p}D"] = values

    return result

def build_shift_matrix(df, series_func):
    base_series = series_func(df)

    result = pd.DataFrame(index=df.index)

    for p in periods:
        result[f"{p}D"] = base_series.shift(p - 1)

    return result

def build_poc_matrix(df, func):
    result = pd.DataFrame(index=df.index)

    for p in [20, 60, 120, 160, 200, 250]:
        values = []

        for i in range(len(df)):
            if i < p - 1:
                values.append(np.nan)
            else:
                window = df.iloc[i-p+1:i+1]
                values.append(func(window))

        result[f"{p}D"] = values

    return result

def build_block_avg(df, series_func, block_size=10, max_days=200):
    series = series_func(df)

    result = pd.DataFrame(index=df.index)

    blocks = list(range(0, max_days, block_size))  # 0-10,10-20,...

    for b in blocks:
        col_name = f"{b+1}-{b+block_size}D"

        values = []

        for i in range(len(df)):
            if i < b + block_size:
                values.append(np.nan)
            else:
                window = series.iloc[i - (b + block_size) + 1 : i - b + 1]
                values.append(window.mean())

        result[col_name] = values

    return result

# =====================================================
# COLOR LOGIC (TEXT ONLY)
# =====================================================
def build_color_matrix(matrix):
    color_df = pd.DataFrame("", index=matrix.index, columns=matrix.columns)

    for col in matrix.columns:
        for i in range(len(matrix)):

            # ❌ SAFE GUARD: skip if 1D does not exist
            if "1D" not in matrix.columns:
                continue

            one_day_value = matrix.iloc[i]["1D"]
            compare_value = matrix.iloc[i][col]

            if pd.isna(one_day_value) or pd.isna(compare_value):
                continue

            if col == "1D":
                continue

            if compare_value > one_day_value:
                color_df.iloc[i, color_df.columns.get_loc(col)] = "color:#00ff88;"
            else:
                color_df.iloc[i, color_df.columns.get_loc(col)] = "color:#ff5555;"

    return color_df

def build_block_color_matrix(matrix):
    color_df = pd.DataFrame("", index=matrix.index, columns=matrix.columns)

    if "1-10D" not in matrix.columns:
        return color_df

    base_col = "1-10D"

    for col in matrix.columns:
        if col == base_col:
            continue

        for i in range(len(matrix)):
            base_val = matrix.iloc[i][base_col]
            comp_val = matrix.iloc[i][col]

            if pd.isna(base_val) or pd.isna(comp_val):
                continue

            if comp_val > base_val:
                color_df.iloc[i, color_df.columns.get_loc(col)] = "color:#00ff88;"
            else:
                color_df.iloc[i, color_df.columns.get_loc(col)] = "color:#ff5555;"

    return color_df

# =====================================================
# DISPLAY FUNCTION
# =====================================================
def render_tab(df, func):
    matrix = build_matrix(df, func)

    display = matrix.tail(20).copy()
    display.index = display.index.strftime("%Y-%m-%d")
    display = display.sort_index(ascending=False)

    color_df = build_color_matrix(display)

    styled = display.style.format("{:.2f}").apply(lambda _: color_df, axis=None)

    st.dataframe(styled, height=600, width="stretch")

# =====================================================
# GUIDE
# =====================================================
def render_guide_tab():
    st.title("📘 System Guide")

    if name == "Guide":

        st.markdown("""
    # 📘 PRICING HEATMAP SYSTEM — USER MANUAL

    This dashboard is designed to compare **market behavior across time blocks**, not to read single candles.

    ---

    # 1️⃣ HOW DATA IS STRUCTURED

    The system uses daily market data:

    - Each row = 1 trading day
    - Columns = Open, High, Low, Close, Volume

    Then the system groups data into **10-day blocks**:

    - 1–10D = most recent 10 trading days
    - 11–20D = previous 10 trading days
    - 21–30D = earlier 10 trading days
    - continues up to 200D

    👉 Each block represents a MARKET REGIME, not individual days.

    ---

    # 2️⃣ HOW EACH TAB IS CALCULATED

    # 🔹 OPEN
    - Uses the first trading price of each block period
    - Represents starting sentiment of that regime

    ---

    # 🔹 HIGH
    - Uses the highest price inside each 10-day block
    - Represents strongest buyer level in that period

    ---

    # 🔹 LOW
    - Uses the lowest price inside each 10-day block
    - Represents strongest seller level in that period

    ---

    # 🔹 CLOSE
    - Uses the last closing price inside each block
    - Represents final market consensus of that period

    ---

    # 3️⃣ STRUCTURE TABS (HH / LL)

    # 🔹 HH (Higher High)
    - Highest price of each block
    - Used to compare whether momentum is increasing across time

    # 🔹 LL (Lower Low)
    - Lowest price of each block
    - Used to detect breakdown or weakening structure

    👉 These are NOT single-day values  
    👉 They are block-level structure levels

    ---

    # 4️⃣ VOLATILITY TABS (ADR / ATR)

    ## 🔹 ADR (Average Daily Range)

    - Measures normal daily movement

    How it is built:
    - Step 1: calculate (High - Low) for every day
    - Step 2: average it inside each 10-day block

    👉 Result: shows how much the market is moving per regime

    ---

    # 🔹 ATR (Average True Range)

    - Measures REAL volatility including gaps

    How it is built:
    - Step 1: include:
      - High - Low
      - High vs previous close
      - Low vs previous close
    - Step 2: average values per block

    👉 Result: shows TRUE risk level of each regime

    ---

    # 5️⃣ TREND TABS (KAMA / EMA)

    # 🔹 EMA
    - Simple trend baseline
    - Smooth average of closing prices

    # 🔹 KAMA
    - Adaptive moving average
    - Reacts faster in trends
    - Slows in sideways markets

    👉 System uses:
    - KAMA vs EMA to detect trend quality

    ---

    # 6️⃣ VWAP (INSTITUTIONAL LEVEL)

    - Volume Weighted Average Price
    - Calculated using 5-minute data

    Meaning:
    - Above VWAP = buyers control session
    - Below VWAP = sellers control session

    👉 This is the “fair value line” of the market

    ---

    # 7️⃣ HOW COLOR CODING WORKS (MOST IMPORTANT)

    ALL block comparisons use ONE reference:

    👉 1–10D block = baseline

    Then:

    # 🟢 GREEN
    - Current block value > 1–10D block
    - Meaning:
      - volatility expanding
      - stronger participation
      - momentum increasing

    ---

    # 🔴 RED
    - Current block value < 1–10D block
    - Meaning:
      - volatility shrinking
      - weaker participation
      - consolidation phase

    ---

    # 8️⃣ HOW TO READ THE SYSTEM (REAL USAGE)

    You should NOT read single indicators.

    You must read relationships:

    ---

    # 🟢 STRONG MARKET CONDITION

    - ADR rising across blocks
    - ATR rising across blocks
    - Price above VWAP
    - KAMA above EMA
    - HH structure increasing

    👉 Meaning: trend expansion phase

    ---

    # 🔴 WEAK / CHOP MARKET

    - ADR flat or falling
    - ATR falling
    - Price stuck near VWAP
    - KAMA below EMA
    - HH/LL not progressing

    👉 Meaning: no directional edge

    ---

    # ⚠️ BREAKOUT SETUP

    - ADR compression (low blocks)
    - then sudden expansion
    - ATR starts rising
    - structure breaks HH or LL

    👉 Meaning: regime change happening

    ---

    # 9️⃣ CORE CONCEPT OF THIS SYSTEM

    This system does NOT predict price.

    It detects:

    - volatility expansion vs compression
    - trend strength vs weakness
    - structural shift across time
    - institutional activity zones

    ---

    # 🔥 FINAL RULE

    👉 Always compare EVERYTHING to 1–10D block  
    That is your system’s anchor reference

    """)
    
# =====================================================
# 🔥 DISPLAY KPI METRICS (TOP PANEL)
# =====================================================
# Update the labels to clarify it is live data
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Open", f"{open_1d:.2f}")
c2.metric("High", f"{high_1d:.2f}")
c3.metric("Low", f"{low_1d:.2f}")
c4.metric("Close", f"{curr_p:.2f}") 
c5.metric("VWAP", f"{vwap_1d:.2f}")
c6.metric("KAMA", f"{kama_1d:.2f}")
c7.metric("POC 3-Month", f"{poc_3m:.2f}")
c8.metric("POC 6-Month", f"{poc_6m:.2f}")

c9, c10, c11, c12, c13, c14, c15, c16 = st.columns(8)
c9.metric("ATR-1D", f"{atr_1d:.2f}")
c10.metric("ATR-5D", f"{atr_5d:.2f}")
c11.metric("ATR-10D", f"{atr_10d:.2f}")
c12.metric("ADR-1D", f"{adr1:.2f}")
c13.metric("ADR-5D", f"{adr5:.2f}")
c14.metric("ADR-10D", f"{adr10:.2f}")
c15.metric("Min ADR-5D", f"{minadr5:.2f}")
c16.metric("Min ADR-10D", f"{minadr10:.2f}")

c1, c2, c3, c4, c5 = st.columns(5)
timeframes = [
    ("05-10DAY", 5, 10, c1),
    ("10-20DAY", 10, 20, c2),
    ("20–50DAY", 20, 50, c3),
    ("50–120DAY", 50, 120, c4),
    ("50–200DAY", 50, 200, c5)
]

for label, start, end, col in timeframes:
    flags = build_flags(df, start, end)
    with col:
        # We use the new helper function here
        st.markdown(market_regime_box(label, flags), unsafe_allow_html=True)

st.divider()

# =====================================================
# TABS
# =====================================================
tab_names = [
    "Close","Open","High","Low","HH","LL","ADR-1D","ADR-10D(AVG)","ATR-1D","ATR-10D(AVG)","KAMA","VWAP","MA","EMA","POC","Guide"]

metric_map = {
    "Close": metric_close,
    "Open": metric_open,
    "High": metric_high,
    "Low": metric_low,
    "MA": metric_ma,
    "VWAP": metric_vwap,
    "HH": metric_hh,
    "LL": metric_ll,    
}

tabs = st.tabs(tab_names)

for tab, name in zip(tabs, tab_names):
    with tab:
        if name == "Guide":
            render_guide_tab()

        elif name == "ADR-1D":
            matrix = build_shift_matrix(df, adr_series)

        elif name == "ATR-1D":
            matrix = build_shift_matrix(df, atr_series)

        elif name == "EMA":
            matrix = build_shift_matrix(df, ema_series)

        elif name == "KAMA":
            matrix = build_shift_matrix(df, kama_series)

        elif name == "POC":
            matrix = build_poc_matrix(df, metric_poc)
            
        elif name == "ADR-10D(AVG)":
            matrix = build_block_avg(df, adr_series, block_size=10, max_days=200)

            display = matrix.tail(20).copy()
            display.index = display.index.strftime("%Y-%m-%d")
            display = display.sort_index(ascending=False)

            color_df = build_block_color_matrix(display)

            styled = display.style.format("{:.2f}").apply(lambda _: color_df, axis=None)

            st.dataframe(styled, height=600, width="stretch")

        elif name == "ATR-10D(AVG)":
            matrix = build_block_avg(df, atr_series, block_size=10, max_days=200)

            display = matrix.tail(20).copy()
            display.index = display.index.strftime("%Y-%m-%d")
            display = display.sort_index(ascending=False)

            color_df = build_block_color_matrix(display)

            styled = display.style.format("{:.2f}").apply(lambda _: color_df, axis=None)

            st.dataframe(styled, height=600, width="stretch")

        else:
            matrix = build_matrix(df, metric_map[name])

        display = matrix.tail(20).copy()
        display.index = display.index.strftime("%Y-%m-%d")
        display = display.sort_index(ascending=False)

        color_df = build_color_matrix(display)
        styled = display.style.format("{:.2f}").apply(lambda _: color_df, axis=None)

        st.dataframe(styled, height=600, width="stretch")

