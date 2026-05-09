import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# =============================
# GLOBAL SETTINGS
# =============================
# This defines the "10 Day" lookback for all indicators
LOOKBACK_DAYS = 10 
# Bars per 16h trading day (Pre+Post)
BARS_PER_DAY_30M = 32  
BARS_PER_DAY_1H = 16

# =============================
# UI CONFIG
# =============================
st.set_page_config(layout="wide", page_title="Trend ABC")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
        .summary-box { background-color: #111; border: 1px solid #333; padding: 10px; border-radius: 5px; text-align: center; }
        .summary-label { color: #888; font-size: 12px; margin-bottom: 2px; }
        .summary-value { color: #00ff00; font-size: 18px; font-weight: bold; margin-bottom: 0px; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🏷 ABC Scanner + WVAD Volume Pulse</h1>", unsafe_allow_html=True)

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

# =============================
# DATA ENGINE
# =============================
def get_data(ticker, interval):
    # Calculate exact bars for the 10-day alignment
    if interval == "30m":
        window = LOOKBACK_DAYS * BARS_PER_DAY_30M
    else:
        window = LOOKBACK_DAYS * BARS_PER_DAY_1H

    # Download 1: display (10d) | Download 2: math buffer (20d)
    df_price = yf.download(ticker, period="10d", interval=interval, prepost=True, auto_adjust=True)
    df_math = yf.download(ticker, period="20d", interval=interval, prepost=True, auto_adjust=True)
    
    if df_price.empty or df_math.empty: return None

    for d in [df_price, df_math]:
        if isinstance(d.columns, pd.MultiIndex): 
            d.columns = d.columns.get_level_values(0)

    # --- WVAD (10-Day Aligned) ---
    hl_range = (df_math['High'] - df_math['Low']).replace(0, 0.0001)
    raw_wvad = ((df_math['Close'] - df_math['Open']) / hl_range) * df_math['Volume']
    df_math['WVAD'] = raw_wvad.rolling(window=window).sum()

    # --- CCI (10-Day Aligned) ---
    tp = (df_math['High'] + df_math['Low'] + df_math['Close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df_math['CCI'] = (tp - sma) / (0.015 * mad)

    # Sync math back to the 10-day price chart
    df_price['WVAD'] = df_math['WVAD']
    df_price['CCI'] = df_math['CCI']
    
    return df_price.astype(float).fillna(0)

def find_recursive_chain(df):
    close = df['Close'].values
    times = df.index
    patterns = []
    cursor = 2
    while cursor < len(close) - 1:
        A, B, C = None, None, None
        for i in range(cursor, len(close) - 1):
            if close[i] < close[i-1] and close[i] < close[i+1]:
                A = {'time': times[i], 'price': close[i], 'idx': i}
                break
        if not A: break
        found_reset = False
        for j in range(A['idx'] + 1, len(close)):
            if close[j] < A['price']:
                cursor = j
                found_reset = True
                break
            if j < len(close) - 1:
                if close[j] > close[j-1] and close[j] > close[j+1]:
                    B = {'time': times[j], 'price': close[j], 'idx': j}
                    break
        if found_reset: continue
        if not B:
            patterns.append({'A': A, 'B': None, 'C': None, 'state': 'SEARCHING_FOR_B'})
            break
        for k in range(B['idx'] + 1, len(close)):
            if close[k] < A['price']:
                cursor = k
                found_reset = True
                break
            if k < len(close) - 1:
                if close[k] < close[k-1] and close[k] < close[k+1]:
                    if close[k] > A['price']:
                        C = {'time': times[k], 'price': close[k], 'idx': k}
                        patterns.append({'A': A, 'B': B, 'C': C, 'state': 'CONFIRMED_C'})
                        cursor = k + 1
                        break
                    else:
                        cursor = k
                        found_reset = True
                        break
        if found_reset: continue
        if not C:
            patterns.append({'A': A, 'B': B, 'C': None, 'state': 'WAITING_FOR_C'})
            break
    return patterns

# =============================
# SUMMARY TOP 
# =============================
def render_summary_top(ticker, interval_label):
    # 5m data for Price/VWAP
    df_5m = yf.download(ticker, period="1d", interval="5m", prepost=True, auto_adjust=True)
    # Aligned 10-day data for indicators
    df_current = get_data(ticker, interval_label)

    if df_5m.empty or df_current is None: return

    if isinstance(df_5m.columns, pd.MultiIndex): 
        df_5m.columns = df_5m.columns.get_level_values(0)

    # --- Logic to find the Last Pivot ---
    patterns = find_recursive_chain(df_current)
    last_pivot_name = "N/A"
    last_pivot_price = 0.0
    
    if patterns:
        last_p = patterns[-1]
        # Determine if the latest point in the last pattern is C, B, or A
        if last_p['C']:
            last_pivot_name, last_pivot_price = "C", last_p['C']['price']
        elif last_p['B']:
            last_pivot_name, last_pivot_price = "B", last_p['B']['price']
        elif last_p['A']:
            last_pivot_name, last_pivot_price = "A", last_p['A']['price']

    # Real-time Price & VWAP
    price_now = df_5m['Close'].iloc[-1]
    vwap = (df_5m['Close'] * df_5m['Volume']).cumsum() / df_5m['Volume'].cumsum()
    last_vwap = vwap.iloc[-1]

    # Aligned Indicators
    last_cci = df_current['CCI'].iloc[-1]
    current_wvad = df_current['WVAD'].iloc[-1]
    wvad_color = "#00ff00" if current_wvad > 0 else "#ff4b4b"

    # UI DISPLAY - Increased to 5 columns
    c1, c2, c3, c4, c5 = st.columns(5)    
    c1.markdown(f"<div class='summary-box'><p class='summary-label'>PRICE NOW (5min)</p><p class='summary-value'>{price_now:.2f}</p></div>", unsafe_allow_html=True)    
    c2.markdown(f"<div class='summary-box'><p class='summary-label'>VWAP (5min)</p><p class='summary-value'>{last_vwap:.2f}</p></div>", unsafe_allow_html=True)    
    c3.markdown(f"<div class='summary-box'><p class='summary-label'>CCI (10-DAY)</p><p class='summary-value'>{last_cci:.2f}</p></div>", unsafe_allow_html=True)    
    c4.markdown(f"<div class='summary-box'><p class='summary-label'>WVAD (10-DAY)</p><p class='summary-value' style='color:{wvad_color}'>{current_wvad:,.0f}</p></div>", unsafe_allow_html=True)
    c5.markdown(f"""
        <div class='summary-box'>
            <p class='summary-label'>LAST PIVOT ({last_pivot_name})</p>
            <p class='summary-value' style='color:#00FF00'>{last_pivot_price:.2f}</p>
        </div>
    """, unsafe_allow_html=True)
    
# =============================
# CHARTING
# =============================
def render_analysis(label, interval):
    # Summary on top of each tab
    render_summary_top(ticker, interval)
    
    df = get_data(ticker, interval)
    if df is None: return
    
    # 1. Strip timezone and create string labels for the Category axis
    df.index = df.index.tz_localize(None)
    df_str_index = df.index.strftime('%Y-%m-%d %H:%M:%S')
    
    patterns = find_recursive_chain(df)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    # 1. PRICE LINE - Use strings for X to match the category axis
    fig.add_trace(go.Scatter(x=df_str_index, y=df['Close'], name="Close Price", line=dict(color="steelblue", width=1.5)), row=1, col=1)
    
    color_set = ["magenta", "yellow", "red"]
    for i, p in enumerate(patterns):
        is_latest = (i == len(patterns) - 1)
        current_color = "#00FF00" if is_latest else color_set[i % 3]
        
        # FIX: Convert pattern timestamps to strings matching the axis
        pts = []
        if p['A']: pts.append({'time': p['A']['time'].strftime('%Y-%m-%d %H:%M:%S'), 'price': p['A']['price'], 'lbl': 'A'})
        if p['B']: pts.append({'time': p['B']['time'].strftime('%Y-%m-%d %H:%M:%S'), 'price': p['B']['price'], 'lbl': 'B'})
        if p['C']: pts.append({'time': p['C']['time'].strftime('%Y-%m-%d %H:%M:%S'), 'price': p['C']['price'], 'lbl': 'C'})

        # Draw lines using string coordinates
        fig.add_trace(go.Scatter(
            x=[pt['time'] for pt in pts], 
            y=[pt['price'] for pt in pts], 
            mode="lines", 
            line=dict(color=current_color, width=1, dash="dot"), 
            showlegend=False
        ), row=1, col=1)
        
        for pt in pts:
            y_offset = -30 if pt['lbl'] == "B" else 30 
            fig.add_annotation(x=pt['time'], y=pt['price'], text=pt['lbl'], showarrow=True, arrowhead=0, arrowwidth=0.5, arrowcolor=current_color, ax=0, ay=y_offset, font=dict(color=current_color, size=10), bgcolor="rgba(0,0,0,0)", row=1, col=1)

    # 2. WVAD - Use strings for X
    fig.add_trace(go.Scatter(x=df_str_index, y=df['WVAD'], name="WVAD", line=dict(color="#FFA500", width=0.8), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.15)', showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color="white", opacity=0.3, row=2, col=1)
    
    # Format the bottom labels
    short_dates = df.index.strftime('%b %d %H:%M')
    fig.update_xaxes(type='category', showgrid=False, tickmode='array', tickvals=df_str_index[::max(1, len(df)//10)], ticktext=short_dates[::max(1, len(df)//10)])
    fig.update_layout(template="plotly_dark", height=480, margin=dict(l=0, r=0, t=40, b=40), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    
    st.plotly_chart(
        fig, 
        width="stretch",
        key=f"chart_{label}",
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

# =============================
# MAIN
# =============================
t1, t2 = st.tabs(["30 MINUTE SCAN", "ONE HOUR SCAN"])
with t1: 
    render_analysis("30m", "30m")
with t2: 
    render_analysis("1h", "1h")
