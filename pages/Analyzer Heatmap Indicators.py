import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# PAGE CONFIG & COMPACT CSS
# =====================================================
st.set_page_config(layout="wide", page_title="Heatmap Indicators")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        html, body, [class*="css"] { font-size: 13px !important; }
        h1 { font-size: 20px !important; margin-bottom: 10px !important; }
        [data-testid="stMetricValue"] { font-size: 15px !important; font-weight: 700; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 13px !important; }
        .sentiment-box {
            background-color: #161a24; padding: 12px; border-radius: 4px;
            border: 1px solid #333; margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🌅 Indicators Heatmap</h1>", unsafe_allow_html=True)

# =====================================================
# MATPLOTLIB THEME
# =====================================================
plt.rcParams.update({
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "legend.fontsize": 7, "figure.facecolor": "#0e1117", "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#444", "xtick.color": "#888", "ytick.color": "#888",
    "grid.color": "#222", "text.color": "white"
})

# =====================================================
# INDICATOR FUNCTIONS
# =====================================================
def calc_rsi(series, period):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calc_mfi(df, period):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    mf = tp * df['Volume']

    direction = tp.diff()

    pos_mf = mf.where(direction > 0, 0)
    neg_mf = mf.where(direction < 0, 0)

    pos_sum = pos_mf.ewm(alpha=1/period, adjust=False).mean()
    neg_sum = neg_mf.ewm(alpha=1/period, adjust=False).mean().replace(0, 1e-9)

    ratio = pos_sum / neg_sum
    return 100 - (100 / (1 + ratio))

def calc_obv(df):
    return (np.sign(df['Close'].diff()).fillna(0) * df['Volume']).cumsum()

def calc_cci(df, period):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma = tp.ewm(alpha=1/period, adjust=False).mean()
    dev = (tp - sma).abs()
    mad = dev.ewm(alpha=1/period, adjust=False).mean()
    return (tp - sma) / (0.015 * mad.replace(0, 1e-9))

def calc_kama(price, period=10, fast=2, slow=30):
    # Convert to Series if not already
    price_series = pd.Series(price)
    
    # 1. Efficiency Ratio (ER) Calculation
    change = (price_series - price_series.shift(period)).abs()
    volatility = price_series.diff().abs().rolling(period).sum()
    volatility = volatility.replace(0, np.nan)
    er = change / volatility

    # 2. Smoothing Constant (SC) Calculation
    sc_fast = 2 / (fast + 1)
    sc_slow = 2 / (slow + 1)
    sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2

    # 3. KAMA Calculation using NumPy for speed
    kama = np.zeros(len(price_series))
    
    # --- THE TWEAK FROM FUNCTION 1 ---
    # Calculate the mean of the first 'period' elements
    initial_mean = price_series.iloc[:period].mean()
    kama[:period] = initial_mean 
    # ---------------------------------

    # Convert SC to numpy for faster loop access
    sc_values = sc.values
    price_values = price_series.values

    for i in range(period, len(price_series)):
        sc_val = sc_values[i]
        if np.isnan(sc_val):
            sc_val = 0 # Keeps KAMA flat if there is no movement

        # Recursive formula: KAMA_prev + SC * (Price_curr - KAMA_prev)
        kama[i] = kama[i-1] + sc_val * (price_values[i] - kama[i-1])

    return pd.Series(kama, index=price_series.index)

def calc_kdj(df, n=9):
    low_min = df['Low'].rolling(window=n, min_periods=1).min()
    high_max = df['High'].rolling(window=n, min_periods=1).max()

    rsv = (df['Close'] - low_min) / (high_max - low_min + 1e-9) * 100

    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d

    return k, d, j

def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calc_wvad_raw(df):
    """Calculates the raw Williams' Variable Accumulation/Distribution value."""
    range_diff = (df['High'] - df['Low']).replace(0, 1e-9)
    # Formula: ((Close - Open) / (High - Low)) * Volume
    return (df['Volume'] * (df['Close'] - df['Open'])) / range_diff

def calc_wvad(df, period):
    """Returns the rolling sum of the raw WVAD."""
    raw = calc_wvad_raw(df)
    return raw.rolling(window=period).sum()

def calc_tor_raw(df):
    # We will use this to compare Volume vs. its own Moving Average
    return df['Volume']

# =====================================================
# UI & DATA
# =====================================================
NY_TZ = ZoneInfo("America/New_York")
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

col1, col_refresh = st.columns([2, 1])

with col1:
    st.markdown(
    f"<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>"
    f"Enter Stock Ticker  |  Last refresh (AMS / NY) : "
    f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
    )
    ticker = st.text_input("Ticker:", value="APP", autocomplete="off", label_visibility="collapsed").strip().upper()
    
with col_refresh:
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    if st.button("🔄"):
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.cache_data.clear()
        st.rerun()    

if ticker:
    df_raw = yf.download(ticker, period="1y", interval="1d", progress=False)
    df_raw = df_raw.sort_index()  # keep ascending for indicators
    df_raw = df_raw[~df_raw.index.duplicated(keep='last')]  # remove duplicate dates
    df_raw = df_raw.sort_index()

    IND = pd.DataFrame(index=df_raw.index)

    close = df_raw['Close']
    high = df_raw['High']
    low = df_raw['Low']
    volume = df_raw['Volume']
    if not df_raw.empty:
        if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)

        # =====================================================
        # DATA PROCESSING WITH YOUR ORIGINAL LOGIC
        # =====================================================
        def get_complete_data(df, p1, p2):
            close = df['Close']
            last_p = close.iloc[-1]
            kama_l, kama_s = calc_kama(close, p2), calc_kama(close, p1)
            rsi_s, rsi_l = calc_rsi(close, p1), calc_rsi(close, p2)
            mfi_s, mfi_l = calc_mfi(df, p1), calc_mfi(df, p2)
    
            v_col = "#00ff88" if v_diff > 0.1 else "#ff3333" if v_diff < -0.1 else "#888"
        
            res = {
                "R_Low": last_p - dollar_move, "R_High": last_p + dollar_move,
                "Sentiment": v, "Color": c, "Score": score, "Bias%": bias_val,
                "RSI": rsi_s.iloc[-1], "MFI": mfi_s.iloc[-1], 
                "Mom%": (close.pct_change(p1).iloc[-1]) * 100,
                "CCI": calc_cci(df, p1).iloc[-1], 
                "Angle": np.degrees(np.arctan(kama_s.diff(1).iloc[-1]))
            }
            return res, res["Color"] # Returns the dict AND the color string separately

        # =====================================================
        # 🔥 TOP MULTI-TIMEFRAME SENTIMENT BAR
        # =====================================================
        def get_sentiment_summary(df, p1, p2):
            df = df.copy()

            # 1. Existing Indicator Calculations
            df['KAMA_L'] = calc_kama(df['Close'], p2)
            df['RSI_S'], df['RSI_L'] = calc_rsi(df['Close'], p1), calc_rsi(df['Close'], p2)
            df['MFI_S'], df['MFI_L'] = calc_mfi(df, p1), calc_mfi(df, p2)
            
            # 2. Institutional Truth (Gatekeeper) using the new function
            df['WVAD_S'] = calc_wvad(df, p1) 

            df = df.dropna()
            if len(df) < 2: return "⚪ NO EDGE", "#888"

            latest, prev = df.iloc[-1], df.iloc[-2]

            # Your standard scoring logic
            def get_score(a, b):
                d = a - b
                return 2 if d > 5 else 1 if d > 1 else -2 if d < -5 else -1 if d < -1 else 0

            score = get_score(latest['RSI_S'], latest['RSI_L']) + get_score(latest['MFI_S'], latest['MFI_L'])
            kama_diff = latest['KAMA_L'] - prev['KAMA_L']
            bias = ((latest['Close'] - latest['KAMA_L']) / latest['KAMA_L']) * 100
            wvad_val = latest['WVAD_S']

            # --- CO-OPERATIVE LOGIC ---
            if abs(kama_diff) <= 0.02:
                if wvad_val > 0: 
                    return "🔍 SILENT ACCUMULATION", "#00ccff"
                return "⚪ NO EDGE", "#888" # Added the comma and color

            elif kama_diff > 0:
                if bias > 3: 
                    # FIX: Added the color "yellow" as the second return value
                    return "⚠️ CAUTION-PRICE > AVERAGE", "yellow" 
                
                if wvad_val < 0:
                    return "⚠️ HOLLOW STRENGTH", "orange" 
                
                # This part is already correct in your code
                return ("🚀 STRONG TREND UP", "#00ff88") if score >= 2 else ("🟢 STABLE UP", "#aaffaa")

            else: # KAMA Slope is Down
                # The MSFT Recovery Case: Price Down but Volume is Green
                if wvad_val > 0 and score > 0:
                    return "🩹 RECOVERY / DIVERGENCE", "#ff8888"
                    
                if score <= -2 and bias < 0:
                    return "💀 STRONG TREND DOWN", "#ff3333"
                return "🔴 TREND SLOW DOWN", "#ff5555"

        # compute all 3 timeframes
        s1, c1 = get_sentiment_summary(df_raw, 5, 10)
        s2, c2 = get_sentiment_summary(df_raw, 10, 20)
        s3, c3 = get_sentiment_summary(df_raw, 20, 50)
        s4, c4 = get_sentiment_summary(df_raw, 50, 200)

        # render UI
        col1, col2, col3, col4 = st.columns(4)

        def sentiment_box(label, value, color):
            return f"""
            <div style="
                background-color:{color};
                padding:10px;
                border-radius:6px;
                text-align:center;
                font-weight:600;
                color:black;">
                <div style="font-size:12px;">{label}</div>
                <div style="font-size:16px;">{value}</div>
            </div>
            """

        with col1:
            st.markdown(sentiment_box("5 / 10", s1, c1), unsafe_allow_html=True)
        with col2:
            st.markdown(sentiment_box("10 / 20", s2, c2), unsafe_allow_html=True)
        with col3:
            st.markdown(sentiment_box("20 / 50", s3, c3), unsafe_allow_html=True)
        with col4:
            st.markdown(sentiment_box("50 / 200", s4, c4), unsafe_allow_html=True)

        tab1, tab2, tab3, tab4, tab5, tab_guide = st.tabs(["⚡ Short-Term (5D/10D)", "Short-Term (10D/20D)", "📈 Mid-Term (20D/50D)", "🌍 Long-Term (50D/200D)", "Day (1D/5D)", "📖 Guide"])
        
        def generate_radar(p1, p2, tab_obj):
            with tab_obj:
                df = df_raw.copy()
                pdf = df.copy()
                close = df['Close']

                # =========================
                # TREND BASE (NO ROLLING MA)
                # =========================
                df['MA_S'] = close.ewm(span=p1, adjust=False).mean()
                df['MA_L'] = close.ewm(span=p2, adjust=False).mean()

                # =========================
                # KAMA (already OK)
                # =========================
                df['KAMA_S'] = calc_kama(close, p1)
                df['KAMA_L'] = calc_kama(close, p2)

                df['K_Ang_S'] = np.degrees(np.arctan(df['KAMA_S'].diff(p1) / p1))
                df['K_Ang_L'] = np.degrees(np.arctan(df['KAMA_L'].diff(p2) / p2))

                # =========================
                # MOMENTUM / OSCILLATORS (OK)
                # =========================
                df['RSI_S'], df['RSI_L'] = calc_rsi(close, p1), calc_rsi(close, p2)
                df['MFI_S'], df['MFI_L'] = calc_mfi(df, p1), calc_mfi(df, p2)

                # =========================
                # OBV (REMOVE ROLLING SMOOTHING)
                # =========================
                df['OBV_RAW'] = calc_obv(df)
                df['OBV_S'] = df['OBV_RAW'].rolling(p1).mean()
                df['OBV_L'] = df['OBV_RAW'].rolling(p2).mean()

                # =========================
                # BIAS (OK)
                # =========================
                df['Bias_S'] = ((close - df['MA_S']) / df['MA_S']) * 100
                df['Bias_L'] = ((close - df['MA_L']) / df['MA_L']) * 100

                # =========================
                # MOMENTUM (OK)
                # =========================
                df['Mom_S'] = close.pct_change(p1) * 100
                df['Mom_L'] = close.pct_change(p2) * 100

                # =========================
                # CCI (OK)
                # =========================
                df['CCI_S'], df['CCI_L'] = calc_cci(df, p1), calc_cci(df, p2)

                # =========================
                # VOLATILITY (REMOVE ROLLING STD)
                # =========================
                log_ret = np.log(close / close.shift(1))
                df['HV_S'] = log_ret.ewm(span=p1, adjust=False).std() * np.sqrt(252) * 100
                df['HV_L'] = log_ret.ewm(span=p2, adjust=False).std() * np.sqrt(252) * 100

                # =========================
                # KDJ / MACD 
                # =========================
                df['K'], df['D'], df['J'] = calc_kdj(df, p1)
                df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = calc_macd(
                    df, fast=p1, slow=p2, signal=int(p2/2)
                )

                # =========================
                # KDJ FILTER 
                # =========================
                df['TREND_UP'] = df['MA_S'] > df['MA_L']
                df['TREND_DOWN'] = df['MA_S'] < df['MA_L']

                kdj_bull = (
                    (df['K'] > df['D']) &
                    (df['J'] > df['K']) &
                    (df['K'] < 80)
                )

                kdj_bear = (
                    (df['K'] < df['D']) &
                    (df['J'] < df['K']) &
                    (df['K'] > 20)
                )

                df['KDJ_BUY'] = kdj_bull & df['TREND_UP']
                df['KDJ_SELL'] = kdj_bear & df['TREND_DOWN']

                # =========================
                # TOR
                # =========================
                # WVAD: Sum of pressure over P1 and P2
                wvad_raw = calc_wvad_raw(df)
                df['WVAD_S'] = (wvad_raw.rolling(window=p1).sum() / 1e6) 
                df['WVAD_L'] = (wvad_raw.rolling(window=p2).sum() / 1e6) 

                # =========================
                # WVAD
                # =========================
                df['TOR_S'] = df['Volume'] / df['Volume'].rolling(window=p1).mean().replace(0, 1e-9)
                df['TOR_L'] = df['Volume'] / df['Volume'].rolling(window=p2).mean().replace(0, 1e-9)

                # =========================
                # CLEAN
                # =========================
                df = df.dropna(subset=['Close'])
                latest = df.iloc[-1]
                prev = df.iloc[-2]

                pdf = df.copy() 

                pdf['WVAD_S'] = pdf['WVAD_S'] / 1e6
                pdf['WVAD_L'] = pdf['WVAD_L'] / 1e6

                # =========================
                # 🔥 LAST 10 DAYS TABLE DATA
                # =========================
                show_cols = [
                    'HV_S', 'HV_L',
                    # Strength
                    'CCI_S', 'CCI_L',
                    # Momentum
                    'MFI_S', 'MFI_L',
                    'Mom_S', 'Mom_L',
                    'RSI_S', 'RSI_L',
                    # Trend
                    'K_Ang_S', 'K_Ang_L',
                    'Bias_S', 'Bias_L',
                    # Volume
                    'OBV_S', 'OBV_L',
                    # MACD
                    'MACD', 'MACD_SIGNAL', 'MACD_HIST',
                    # KDJ
                    'K', 'D', 'J',
                    'TOR_S', 'TOR_L', 'WVAD_S', 'WVAD_L'
                    
                ]

                st.markdown(
                    f"<p style='margin-bottom:10px; color:steelblue; font-size:15px; font-weight:bold;'>📅 LAST 10 TRADING DAYS DATA TABLE</p>", unsafe_allow_html=True)
                st.markdown("<p style='margin-bottom: 0px; color: steelblue; font-size: 15px; font-weight: bold;'>KDJ(Momentum+Timing) | MACD(Momentum+Trend) | HV(Volatility) | CCI-MFI-MOM-RSI(Momentum) | ANGLE-BIAS(Trend) | OBV(Volume)</p>", unsafe_allow_html=True)                
                df_last5 = df[show_cols].tail(20).copy()
                df_last5 = df_last5.sort_index(ascending=False)

                # format date
                df_last5.index = df_last5.index.strftime("%Y-%m-%d")

                # rounding
                df_last5 = df_last5.round(2)

                # compress OBV (millions)
                df_last5['OBV_S'] = (df_last5['OBV_S'] / 1e6).round(2)
                df_last5['OBV_L'] = (df_last5['OBV_L'] / 1e6).round(2)

                # rename MACD AFTER calculations
                df_last5 = df_last5.rename(columns={
                    'MACD': 'DIF',
                    'MACD_SIGNAL': 'DEA',
                    'MACD_HIST': 'MACD'
                })

                # =========================
                # 🔥 HEATMAP ENGINE (FIXED)
                # =========================
                def heat_style(v, low, high):
                    try:
                        v = float(v)
                    except:
                        return ""

                    if v <= low:
                        return "background-color: #8b1a1a; color: white"   # 🔴 RED
                    elif v >= high:
                        return "background-color: #1f7a1f; color: white"   # 🟢 GREEN
                    return ""

                # =========================
                # UPDATED 4-COLOR INDICATOR STYLES
                # =========================
                def apply_4_color(v, red_max, pink_max, l_green_min, green_min):
                    """
                    Helper function to handle the 4-color CSS logic.
                    v: current value
                    red_max: below this is RED
                    pink_max: below this is PINK
                    l_green_min: above this is LIGHT GREEN
                    green_min: above this is GREEN
                    """
                    try:
                        val = float(v)
                    except:
                        return ""

                    if val <= red_max:
                        return "background-color: #FF0000; color: white;"      # RED
                    elif val <= pink_max:
                        return "background-color: #FFC0CB; color: black;"      # PINK
                    elif val >= green_min:
                        return "background-color: #008000; color: white;"      # GREEN
                    elif val >= l_green_min:
                        return "background-color: #90EE90; color: black;"      # LIGHT GREEN
                    else:
                        return "" # Default (Neutral)

                def rsi_style(v, p):
                    # This keeps your 'p' (period) logic but uses the 4-color helper
                    if p <= 10:
                        # For short periods, we use tighter bounds
                        return apply_4_color(v, 20, 40, 60, 80)
                    elif p <= 20:
                        return apply_4_color(v, 20, 40, 60, 80)
                    else:
                        # For longer periods, we use slightly wider bounds
                        return apply_4_color(v, 15, 35, 65, 85)

                def mfi_style(v, p):
                    if p <= 10:
                        return apply_4_color(v, 20, 40, 60, 80)
                    else:
                        return apply_4_color(v, 15, 35, 65, 85)

                def j_style(v, p):
                    if p <= 10:
                        return apply_4_color(v, 10, 30, 70, 90)
                    else:
                        return apply_4_color(v, 5, 25, 75, 95)

                def cci_style(v, p):
                    if p <= 10:
                        return apply_4_color(v, -150, -100, 100, 150)
                    else:
                        return apply_4_color(v, -120, -80, 80, 120)

                def macd_style(v, p):
                    if p <= 10:
                        return apply_4_color(v, -1.0, -0.2, 0.2, 1.0)
                    else:
                        return apply_4_color(v, -3.0, -1.0, 1.0, 3.0)

                def bias_style(v, p):
                    return apply_4_color(v, -3, -1, 1, 3)

                def mom_style(v, p):
                    return apply_4_color(v, -5, -2, 2, 5)

                def angle_style(v, p):
                    return apply_4_color(v, -30, -10, 10, 30)

                def hv_style(v, p):
                    # Volatility is usually positive; we use 4 stages of intensity
                    return apply_4_color(v, 5, 15, 30, 50)

                def tor_style(v, p):
                    # Uses period p to set limits (1.8 for short, 1.5 for long)
                    limit = 1.8 if p <= 10 else 1.5
                    return apply_4_color(v, 0.5, 0.8, 1.2, limit)

                def wvad_style(v, p):
                    limit = 2.0 if p <= 10 else 5.0
                    return apply_4_color(v, -limit, -0.5, 0.5, limit)
                
                def obv_style(v, p):
                    try:
                        v = float(v)
                    except:
                        return ""
                    # OBV remains 2-color as it is usually a trend direction (Positive/Negative)
                    return "background-color: #008000; color: white" if v > 0 else "background-color: #FF0000; color: white"

                # =========================
                # APPLY STYLING
                # =========================
                styled = df_last5.style   # ❗ MUST ONLY ONCE

                styled = styled.map(lambda v: rsi_style(v, p1), subset=["RSI_S"])
                styled = styled.map(lambda v: rsi_style(v, p2), subset=["RSI_L"])

                styled = styled.map(lambda v: mfi_style(v, p1), subset=["MFI_S"])
                styled = styled.map(lambda v: mfi_style(v, p2), subset=["MFI_L"])

                styled = styled.map(lambda v: j_style(v, p1), subset=["J"])

                styled = styled.map(lambda v: cci_style(v, p1), subset=["CCI_S"])
                styled = styled.map(lambda v: cci_style(v, p2), subset=["CCI_L"])

                styled = styled.map(lambda v: macd_style(v, p1), subset=["MACD"])

                styled = styled.map(lambda v: bias_style(v, p1), subset=["Bias_S"])
                styled = styled.map(lambda v: bias_style(v, p2), subset=["Bias_L"])

                styled = styled.map(lambda v: mom_style(v, p1), subset=["Mom_S"])
                styled = styled.map(lambda v: mom_style(v, p2), subset=["Mom_L"])

                styled = styled.map(lambda v: angle_style(v, p1), subset=["K_Ang_S"])
                styled = styled.map(lambda v: angle_style(v, p2), subset=["K_Ang_L"])

                styled = styled.map(lambda v: hv_style(v, p1), subset=["HV_S"])
                styled = styled.map(lambda v: hv_style(v, p2), subset=["HV_L"])

                styled = styled.map(lambda v: obv_style(v, p1), subset=["OBV_S"])
                styled = styled.map(lambda v: obv_style(v, p2), subset=["OBV_L"])

                # TOR (Volume Turnover)
                styled = styled.map(lambda v: tor_style(v, p1), subset=["TOR_S"])
                styled = styled.map(lambda v: tor_style(v, p2), subset=["TOR_L"])

                # WVAD (Institutional Pressure)
                styled = styled.map(lambda v: wvad_style(v, p1), subset=["WVAD_S"])
                styled = styled.map(lambda v: wvad_style(v, p2), subset=["WVAD_L"])

                # =========================
                # DISPLAY (CORRECT)
                # =========================
                st.dataframe(
                    styled.format({
                        'Close': "${:.2f}",
                        'HV_S': "{:.0f}%",
                        'HV_L': "{:.0f}%",
                        'CCI_S': "{:.0f}",
                        'CCI_L': "{:.0f}",
                        'MFI_S': "{:.0f}",
                        'MFI_L': "{:.0f}",
                        'Mom_S': "{:.0f}%",
                        'Mom_L': "{:.0f}%",
                        'RSI_S': "{:.0f}",
                        'RSI_L': "{:.0f}",
                        'K_Ang_S': "{:.0f}",
                        'K_Ang_L': "{:.0f}",
                        'Bias_S': "{:.0f}%",
                        'Bias_L': "{:.0f}%",
                        'OBV_S': "{:.0f}",
                        'OBV_L': "{:.0f}",
                        "K": "{:.0f}",
                        "D": "{:.0f}",
                        "J": "{:.0f}",
                        'DIF': "{:.0f}",
                        'DEA': "{:.0f}",
                        'MACD': "{:.0f}",
                        'TOR_S': "{:.2f}x",
                        'TOR_L': "{:.2f}x",
                        'WVAD_S': "{:.0f}M",
                        'WVAD_L': "{:.0f}M"
                    }),
                    width="stretch",
                    height=330
                )
                
                # CHARTS
                pdf = df.copy().tail(60)
                #pdf = df.copy()
    
                x = np.arange(len(pdf))
                xl = pdf.index.strftime("%m%d")

                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.01,
                    row_heights=[0.4, 0.3, 0.3]
                )

                # =========================
                # PRICE
                # =========================
                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['Close'],
                    name=f"CLOSE: ${latest['Close']:.2f}",
                    line=dict(color='steelblue', width=1)
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['MA_S'],
                    name=f"MA{p1}",
                    line=dict(color='white', dash='dot', width=1)
                ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['MA_L'],
                    name=f"MA{p2}",
                    line=dict(color='lightgreen', width=1)
                ), row=1, col=1)

                # =========================
                # KDJ
                # =========================
                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['K'],
                    name=f"K(FAST-OB(90) / OS(10)): {latest['K']:.2f}",
                    line=dict(width=1)
                ), row=2, col=1)

                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['D'],
                    name=f"D(SLOW-OB(80 ) / OS(20): {latest['D']:.2f}",
                    line=dict(width=1, color='orange')
                ), row=2, col=1)

                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['J'],
                    name=f"J (Direction): {latest['J']:.2f}",
                    line=dict(width=1,color='white')
                ), row=2, col=1)

                # Overbought/Oversold lines
                fig.add_hline(y=80, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=20, line_dash="dot", line_color="green", row=2, col=1)

                # =========================
                # MACD
                # =========================
                colors = ['green' if v >= 0 else 'red' for v in pdf['MACD_HIST']]

                fig.add_trace(go.Bar(
                    x=pdf.index,
                    y=pdf['MACD_HIST'],
                    marker_color=colors,
                    name='MACD'
                ), row=3, col=1)

                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['MACD'],
                    name=f"DIF (FAST): {latest['MACD']:.0f}",
                    line=dict(width=1)
                ), row=3, col=1)

                fig.add_trace(go.Scatter(
                    x=pdf.index, y=pdf['MACD_SIGNAL'],
                    name=f"DEA (SLOW): {latest['MACD_SIGNAL']:.0f}",
                    line=dict(width=1, color="white")
                ), row=3, col=1)

                # =========================
                # LAYOUT
                # =========================
                fig.update_layout(
                    height=900,
                    hovermode="x unified",  
                    template="plotly_dark",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0
                    ),
                    font=dict(size=10)  
                )

                st.plotly_chart(fig, width='stretch')

                cl, cr = st.columns(2)
                with cl:
                    f1, a1 = plt.subplots(5, 1, figsize=(5, 13), sharex=True)
                    # HV
                    a1[0].plot(x, pdf['HV_S'], color='white', ls='--', lw=0.5)
                    a1[0].plot(x, pdf['HV_L'], color='lightgreen', lw=0.5)
                    a1[0].legend([f"HV{p1}: {latest[f'HV_S']:.2f}%", f"HV{p2}: {latest[f'HV_L']:.2f}%"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    # CCI
                    a1[1].plot(x, pdf['CCI_S'], color='white', ls='--', lw=0.5)
                    a1[1].plot(x, pdf['CCI_L'], color='lightgreen', lw=0.5)
                    a1[1].axhline(0, color='silver', ls=':', lw=0.5);
                    a1[1].legend([f"CCI{p1}: {latest[f'CCI_S']:.2f}", f"CCI{p2}: {latest[f'CCI_L']:.2f}"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    # MOM
                    a1[2].plot(x, pdf['Mom_S'], color='white', ls='--', lw=0.5)
                    a1[2].plot(x, pdf['Mom_L'], color='lightgreen', lw=0.5)
                    a1[2].axhline(0, color='silver', ls=':', lw=0.5);
                    a1[2].legend([f"MOM{p1}: {latest[f'Mom_S']:.2f}%", f"MOM{p2}: {latest[f'Mom_L']:.2f}%"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    # ANGLE
                    a1[3].plot(x, pdf['K_Ang_S'], color='white', ls='--', lw=0.5)
                    a1[3].plot(x, pdf['K_Ang_L'], color='lightgreen', lw=0.5)
                    a1[3].axhline(0, color='silver', ls=':', lw=0.5);
                    a1[3].legend([f"Ang{p1}: {latest[f'K_Ang_S']:.2f}°", f"Ang{p2}: {latest[f'K_Ang_L']:.2f}°"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    # OBV
                    a1[4].plot(x, pdf['OBV_S'], color='white', ls='--', lw=0.5)
                    a1[4].plot(x, pdf['OBV_L'], color='lightgreen', lw=0.5)
                    a1[4].legend([f"OBV{p1}: {latest[f'OBV_S']/1e6:.2f}M", f"OBV{p2}: {latest[f'OBV_L']/1e6:.2f}M"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    for ax in a1: ax.grid(True, alpha=0.1); ax.axvline(x[-1], color='white', ls=':', alpha=0.3)
                    for ax in a1:
                        ax.set_xticks(x[::4])
                        ax.set_xticklabels(xl[::4])
                        ax.tick_params(axis='x', labelbottom=True)
                    for ax in a1:
                        ax.tick_params(axis='x', colors='white', labelsize=4)
                        ax.tick_params(axis='y', colors='white', labelsize=5)
                    st.pyplot(f1)

                with cr:
                    f2, a2 = plt.subplots(5, 1, figsize=(5, 12.85), sharex=True)
                    # TOR
                    a2[0].plot(x, pdf['TOR_S'], color='white', ls='--', lw=0.5)
                    a2[0].plot(x, pdf['TOR_L'], color='lightgreen', lw=0.5)
                    a2[0].axhline(1.0, color='silver', ls=':', lw=0.5)
                    a2[0].legend([f"TOR{p1}: {latest['TOR_S']:.2f}x", f"TOR{p2}: {latest['TOR_L']:.2f}x"], 
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)                    
                    # MFI
                    a2[1].plot(x, pdf['MFI_S'], color='white', ls='--', lw=0.5)
                    a2[1].plot(x, pdf['MFI_L'], color='lightgreen', lw=0.5)
                    a2[1].axhline(80, color='lime', ls=':', lw=0.5);
                    a2[1].axhline(50, color='silver', ls=':', lw=0.5);
                    a2[1].axhline(20, color='red', ls=':', lw=0.5);
                    a2[1].legend([f"MFI{p1}: {latest[f'MFI_S']:.2f}", f"MFI{p2}: {latest[f'MFI_L']:.2f}"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    # RSI
                    a2[2].plot(x, pdf['RSI_S'], color='white', ls='--', lw=0.5)
                    a2[2].plot(x, pdf['RSI_L'], color='lightgreen', lw=0.5)
                    a2[2].axhline(80, color='lime', ls=':', lw=0.5);
                    a2[2].axhline(50, color='silver', ls=':', lw=0.5);
                    a2[2].axhline(20, color='red', ls=':', lw=0.5);
                    a2[2].legend([f"RSI{p1}: {latest[f'RSI_S']:.2f}", f"RSI{p2}: {latest[f'RSI_L']:.2f}"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    # BIAS
                    a2[3].plot(x, pdf['Bias_S'], color='white', ls='--', lw=0.5)
                    a2[3].plot(x, pdf['Bias_L'], color='lightgreen', lw=0.5)
                    a2[3].axhline(0, color='silver', ls=':', lw=0.5);
                    a2[3].legend([f"Bias{p1}: {latest['Bias_S']:.2f}%", f"Bias{p2}: {latest['Bias_L']:.2f}%"],
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    # WVAD
                    a2[4].plot(x, pdf['WVAD_S'], color='white', ls='--', lw=0.5)
                    a2[4].plot(x, pdf['WVAD_L'], color='lightgreen', lw=0.5)      
                    a2[4].axhline(0, color='silver', ls=':', lw=0.5)
                    a2[4].legend([f"WVAD{p1}: {latest['WVAD_S']/1e6:.2f}M", f"WVAD{p2}: {latest['WVAD_L']/1e6:.2f}M"], 
                                  loc='upper center', bbox_to_anchor=(0.5, 1), fontsize=5, ncol=2, frameon=False)
                    for ax in a2: ax.grid(True, alpha=0.1); ax.axvline(x[-1], color='white', ls=':', alpha=0.3)
                    for ax in a2:
                        ax.set_xticks(x[::4])
                        ax.set_xticklabels(xl[::4])
                        ax.tick_params(axis='x', labelbottom=True)
                    for ax in a2:
                        ax.tick_params(axis='x', colors='white', labelsize=4)
                        ax.tick_params(axis='y', colors='white', labelsize=5)
                    st.pyplot(f2)
        
        generate_radar(5, 10, tab1)
        generate_radar(10, 20, tab2)
        generate_radar(20, 50, tab3)
        generate_radar(50, 200, tab4)
        generate_radar(1, 5, tab5)

        with tab_guide:
            st.markdown("""
            ## 📖 Indicator Radar Guide
            ### 🟢 Bullish Scenarios
            * **🚀 STRONG TREND UP**: High momentum (Score ≥ 2) + healthy Bias (0-3%).
            * **💰 CATCHING UP**: Trend is Up, but price is pulling back to the average (Bias ≤ 1%).
            * **⚠️ CAUTION MONITOR**: Overextended rubber band (Bias > 3%). Pullback likely.
            ### 🔴 Bearish Scenarios
            * **💀 STRONG TREND DOWN**: Price and momentum both diving (Score ≤ -2).
            * **🩹 REVERSAL MONITOR**: Price is down, but momentum is starting to lead up (Score > 0).
            ### 📊 Score Card
            * **+4 to +2**: Short-term strength leading the long-term trend.
            * **-2 to -4**: Short-term weakness leading the breakdown.
            """)
