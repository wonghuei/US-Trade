import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# =============================
# UI CONFIG
# =============================
st.set_page_config(layout="wide", page_title="Trend ABC")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
        [data-testid="stMetricValue"] { font-size: 15px !important; font-weight: 700; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 13px !important; }
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
    df = yf.download(ticker, period="7d", interval=interval, auto_adjust=True)
    if df.empty: return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # WVAD Calculation
    # Formula: ((Close - Open) / (High - Low)) * Volume
    # We sum it over 14 periods to show the accumulation trend
    hl_range = (df['High'] - df['Low']).replace(0, 0.0001) # Avoid div by zero
    raw_wvad = ((df['Close'] - df['Open']) / hl_range) * df['Volume']
    df['WVAD'] = raw_wvad.rolling(window=14).sum()
    
    return df.astype(float).fillna(0)

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
# CHARTING
# =============================
def render_analysis(label, interval):
    df = get_data(ticker, interval)
    if df is None: return
    patterns = find_recursive_chain(df)
    
    # Create Subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3]
    )

    # 1. PRICE LINE (Row 1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name="Close Price", 
        line=dict(color="steelblue", width=1.5), showlegend=True
    ), row=1, col=1)

    color_set = ["magenta", "yellow", "red"]

    # 2. ABC CHAINS (Row 1)
    for i, p in enumerate(patterns):
        is_latest = (i == len(patterns) - 1)
        current_color = "#00FF00" if is_latest else color_set[i % 3]
        pts = [p['A']]
        labels = ["A"]
        if p['B']: 
            pts.append(p['B'])
            labels.append("B")
        if p['C']: 
            pts.append(p['C'])
            labels.append("C")

        # Connection Lines
        fig.add_trace(go.Scatter(
            x=[pt['time'] for pt in pts], y=[pt['price'] for pt in pts],
            mode="lines", line=dict(color=current_color, width=1, dash="dot"),
            showlegend=False
        ), row=1, col=1)

        # Labels
        for pt, lbl in zip(pts, labels):
            y_offset = -30 if lbl == "B" else 30 
            fig.add_annotation(
                x=pt['time'], y=pt['price'], text=lbl, showarrow=True, arrowhead=0,
                arrowwidth=0.5, arrowcolor=current_color, ax=0, ay=y_offset,
                font=dict(color=current_color, size=10, family="Arial"),
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", row=1, col=1
            )

    # 3. WVAD SUB-CHART (Row 2) - NO WEEKENDS
    fig.add_trace(go.Scatter(
        x=df.index, y=df['WVAD'], name="WVAD", 
        line=dict(color="#FFA500", width=0.8),
        fill='tozeroy', 
        fillcolor='rgba(255, 165, 0, 0.15)',
        showlegend=False
    ), row=2, col=1)

    # Add 0-Line for WVAD
    fig.add_hline(y=0, line_width=1, line_color="white", opacity=0.3, row=2, col=1)

    # SHARED AXIS CONFIG
    short_dates = df.index.strftime('%b %d %H:%M')
    
    fig.update_xaxes(
        type='category', # This is what hides weekends and aligns both rows
        showgrid=False,
        tickmode='array',
        tickvals=df.index[::max(1, len(df)//10)], 
        ticktext=short_dates[::max(1, len(df)//10)],
        tickangle=0,
    )

    fig.update_layout(
        template="plotly_dark", height=500,
        yaxis=dict(gridcolor="#333", side="right"),
        #yaxis2=dict(gridcolor="#333", side="right", title="WVAD Vol Pulse"),
        margin=dict(l=0, r=0, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

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

# MAIN
t1, t2 = st.tabs(["15 MINUTE SCAN", "30 MINUTE SCAN"])
with t1: render_analysis("15m", "15m")
with t2: render_analysis("30m", "30m")
