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
st.set_page_config(layout="wide", page_title="ABC")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
        [data-testid="stMetricValue"] { font-size: 15px !important; font-weight: 700; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 13px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🏷 ABC Scanner</h1>", unsafe_allow_html=True)

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
    # period="7d" downloads only active trading days
    df = yf.download(ticker, period="5d", interval=interval, auto_adjust=True)
    if df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # CCI Calculation (Runs only on available trading bars)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(window=14).mean()
    md = tp.rolling(window=14).apply(lambda x: np.fabs(x - x.mean()).mean())
    df['CCI'] = (tp - ma) / (0.015 * md)
    
    return df.astype(float)

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
    
    # SETUP SUBPLOTS
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.7, 0.3]
    )

    # 1. PRICE LINE (Row 1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'], name="Close Price", 
        line=dict(color="steelblue", width=1), showlegend=True
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

        # ABC Labels
        for pt, lbl in zip(pts, labels):
            y_offset = -30 if lbl == "B" else 30 
            fig.add_annotation(
                x=pt['time'], y=pt['price'], text=lbl, showarrow=True, arrowhead=0,
                arrowwidth=0.5, arrowcolor=current_color, ax=0, ay=y_offset,
                font=dict(color=current_color, size=8, family="Arial"),
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", row=1, col=1
            )

        # Transition Line to next A
        if i < len(patterns) - 1:
            next_p = patterns[i+1]
            if p['C'] and next_p['A']:
                next_color = "#00FF00" if (i + 1 == len(patterns) - 1) else color_set[(i + 1) % 3]
                fig.add_trace(go.Scatter(
                    x=[p['C']['time'], next_p['A']['time']],
                    y=[p['C']['price'], next_p['A']['price']],
                    mode="lines", line=dict(color=next_color, width=1, dash="dot"),
                    showlegend=False
                ), row=1, col=1)

        # Trigger Line
        if is_latest and p['B']:
            fig.add_hline(
                y=p['B']['price'], line_dash="dash", line_width=0.8, line_color="#FF4B4B", 
                annotation_text=f"Break B: {p['B']['price']:.2f}", annotation_position="top left",
                row=1, col=1
            )

    # 3. CCI SUB-CHART (Row 2)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['CCI'], name="CCI", 
        line=dict(color="pink", width=0.5), showlegend=False
    ), row=2, col=1)

    # CCI Reference Levels
    fig.add_hline(y=100, line_dash="dot", line_color="red", line_width=0.5, row=2, col=1)
    fig.add_hline(y=-100, line_dash="dot", line_color="green", line_width=1, row=2, col=1)

    # X-AXIS FIX: Force BOTH charts to use 'category' to exclude weekends
    short_dates = df.index.strftime('%b %d %H:%M')
    
    fig.update_xaxes(
        type='category', # CRITICAL: This hides weekends for both rows
        showgrid=False,
        tickmode='array',
        tickvals=df.index[::len(df)//10] if len(df) > 10 else df.index, 
        ticktext=short_dates[::len(df)//10] if len(df) > 10 else short_dates,
        tickangle=0,
    )

    fig.update_layout(
        template="plotly_dark", height=500,
        yaxis=dict(gridcolor="#333", side="right"),
        yaxis2=dict(gridcolor="#333", side="right"),
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
