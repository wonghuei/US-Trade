import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz

# ===============================
# CONFIG & CSS
# ===============================
st.set_page_config(layout="wide", page_title="Price Zone STD")

NY_TZ = pytz.timezone("America/New_York")
MIN_TOUCHES = 3
BASE_TOL = 0.0025

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 18px !important; margin-bottom: 10px !important; }
        [data-testid="stMetricValue"] { font-size: 13px !important; font-weight: 700; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Price Zone (Daily Price) + Volume Profile</h1>", unsafe_allow_html=True)

# ===============================
# ENGINES: KAMA & ZONES
# ===============================

def kama_dynamic(price, lookback=20):
    if not isinstance(price, pd.Series):
        price = pd.Series(price.squeeze())
    returns = price.pct_change().dropna()
    volatility_std = returns.rolling(lookback).std().mean() if not returns.empty else 0
    recent = price.iloc[-lookback:]
    trend_strength = (recent.max() - recent.min()) / recent.mean() if len(price) >= lookback else 0
    n = 5 if volatility_std > 0.02 else 10 if volatility_std > 0.01 else 20
    fast, slow = (2, 20) if trend_strength > 0.05 else (5, 50)
    change = abs(price - price.shift(n))
    vol_sum_k = abs(price.diff()).rolling(window=n).sum()
    er = (change / vol_sum_k).fillna(0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = pd.Series(index=price.index, dtype=float)
    kama.iloc[:n] = price.iloc[:n].rolling(window=n, min_periods=1).mean()
    for i in range(n, len(price)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (price.iloc[i] - kama.iloc[i-1])
    return kama.ffill()

def adaptive_tolerance(df, price, base=BASE_TOL):
    rets = df['Close'].pct_change().dropna()
    vol = float(rets.std() * np.sqrt(252)) if not rets.empty else 0.20
    price_factor = np.clip(price / 100, 0.5, 3.0)
    vol_factor = np.clip(vol / 0.3, 0.5, 2.5)
    mult = (0.5 * price_factor + 0.3 * vol_factor + 0.2)
    return max(base * mult, 0.004)

def calculate_zones(df, price_column, tolerance):
    if df.empty: return pd.DataFrame()
    prices, volumes = df[price_column].values, df['Volume'].values
    zones = []
    for p, v in zip(prices, volumes):
        found = False
        for z in zones:
            if abs(p - z['Zone Mid']) / z['Zone Mid'] <= tolerance:
                z['Touches'] += 1
                z['Prices'].append(float(p))
                z['Total Vol'] += float(v)
                z['Zone Mid'] = float(np.median(z['Prices']))
                found = True; break
        if not found:
            zones.append({'Zone Mid': float(p), 'Touches': 1, 'Prices': [float(p)], 'Total Vol': float(v)})
    
    z_df = pd.DataFrame(zones)
    if z_df.empty: return z_df
    z_df = z_df[z_df['Touches'] >= MIN_TOUCHES].copy()
    z_df['Zone Low'] = z_df['Prices'].apply(min)
    z_df['Zone High'] = z_df['Prices'].apply(max)
    z_df['Vol/Touch'] = z_df['Total Vol'] / z_df['Touches']
    z_df['Touch-Rank'] = (z_df['Touches'] / z_df['Touches'].max()).apply(
        lambda s: "Very Strong" if s >= 0.8 else "Strong" if s >= 0.6 else "Medium" if s >= 0.4 else "Weak")
    z_df['Trade-Rank'] = (z_df['Total Vol'] / z_df['Total Vol'].max()).apply(
        lambda s: "Very Strong" if s >= 0.8 else "Strong" if s >= 0.6 else "Medium" if s >= 0.4 else "Weak")
    z_df['Manip?'] = z_df['Vol/Touch'] > (z_df['Vol/Touch'].median() * 3)
    return z_df.sort_values(by='Zone Mid', ascending=False)

def display_full_table(zones_df, current_price, label, zone_use, adr=None, last_time="N/A"):
    if zones_df.empty: return st.write(f"No {label} zones found.")
    closest_idx = (zones_df['Zone Mid'] - current_price).abs().idxmin()
    html_output = f'''<div style="font-family: 'Courier New', monospace; white-space: pre; background-color: #0E1117; 
                    padding: 15px; border: 1px solid #31333F; border-radius: 8px; font-size: 11px; line-height: 1.1; width: 100%; margin-bottom: 25px; overflow-x: auto;">'''
    html_output += f"<span>--- {label} --- (Now: {current_price:.2f} (at {last_time} EST) | {zone_use} | ADR: {adr:.2f})</span>\n"
    html_output += "<span>" + "-" * 152 + "</span>\n"
    headers = (f"{'Zone Low':>11} {'Zone High':>11} {'Zone Mid':>11} {'Touch':>8} "
               f"{'Touch-Rank':>13} {'Touch-Vol':>18} {'Diff($)':>9} {'Diff(%)':>9} "
               f"{'Trade-Rank':>13} {'Trade-Vol':>15} {'Manip?':>8}\n")
    html_output += f"<span>{headers}</span><span>" + "-" * 152 + "</span>\n"

    for idx, row in zones_df.iterrows():
        diff_dollar = row['Zone Mid'] - current_price
        diff_pct = (diff_dollar / current_price) * 100
        color = "#FFFFFF"
        if row['Touch-Rank'] == "Very Strong": color = "#00FF00"
        elif row['Touch-Rank'] == "Strong": color = "#FFFF00"
        elif row['Touch-Rank'] == "Medium": color = "#00FFFF"
        if idx == closest_idx: color = "#FF00FF"
        manip = "YES" if row['Manip?'] else ""
        line = (f"{row['Zone Low']:>11.2f} {row['Zone High']:>11.2f} {row['Zone Mid']:>11.2f} "
                f"{row['Touches']:>8} {row['Touch-Rank']:>13} {row['Vol/Touch']:>18,.0f} "
                f"{diff_dollar:>9.2f} {diff_pct:>9.2f}% "
                f"{row['Trade-Rank']:>13} {row['Total Vol']:>15,.0f} {manip:>8}\n")
        html_output += f'<span style="color:{color};">{line}</span>'
    st.markdown(html_output + "</div>", unsafe_allow_html=True)

# ===============================
# UI INPUTS
# ===============================
import streamlit as st

col1, col2, col_btn = st.columns([1, 2, 1]) # Adjusted ratios for better fit

with col1:
    ticker = st.text_input("Enter Ticker", value="APP", autocomplete="off").upper()

with col2:
    lookback = st.radio("Historical Range", ["1y", "2y", "5y"], index=0, horizontal=True)

with col_btn:
    # Add vertical space to align with the labels of the other widgets
    st.markdown('<p style="margin-bottom: 24px;"></p>', unsafe_allow_html=True) 
    run_scan = st.button("🔄", use_container_width=True)
    
if ticker:
    df = yf.download(ticker, period=lookback, interval="1d", progress=False)
    df_now = yf.download(ticker, period="1d", interval="5m", prepost=True, progress=False)
    
    if df.empty or df_now.empty: st.stop()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    if isinstance(df_now.columns, pd.MultiIndex): df_now.columns = df_now.columns.get_level_values(0)

    # Core Calculations
    curr_p = float(df_now['Close'].iloc[-1])
    last_time = df_now.index[-1].strftime('%H:%M:%S')
    adr_val = float((df['High'].tail(10) - df['Low'].tail(10)).mean())
    vwap = (df['Close'] * df['Volume']).sum() / df['Volume'].sum()
    df["KAMA"] = kama_dynamic(df["Close"]) # Restored KAMA
    tol = adaptive_tolerance(df, curr_p)

    # PASTE THIS NEW LOGIC HERE:
    counts, bin_edges = np.histogram(df['Close'].values, bins=50, weights=df['Volume'].values)
    max_bin_idx = np.argmax(counts)
    poc_low = bin_edges[max_bin_idx]
    poc_high = bin_edges[max_bin_idx + 1]
    poc_mid = (poc_low + poc_high) / 2
    
    # 52W, 90D, 120D Levels
    h52, l52 = float(df["High"].max()), float(df["Low"].min())
    h30, l30 = float(df["High"].tail(30).max()), float(df["Low"].tail(30).min())
    h60, l60 = float(df["High"].tail(60).max()), float(df["Low"].tail(60).min())
    h90, l90 = float(df["High"].tail(90).max()), float(df["Low"].tail(90).min())
    h120, l120 = float(df["High"].tail(120).max()), float(df["Low"].tail(120).min())

    # Metrics Panel
    r1, r2, r3 = st.columns(3)
    r1.metric("PRICE NOW", f"{curr_p:.2f}")
    r2.metric("VWAP/KAMA", f"{df['KAMA'].iloc[-1]:.2f}")    
    r3.metric("POC Range", f"{poc_low:.2f} - {poc_high:.2f}", f"Mid: {poc_mid:.2f}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("30D H/L", f"{h30:.2f}/{l30:.2f}")
    m2.metric("60D H/L", f"{h60:.2f}/{l60:.2f}")
    m3.metric("90D H/L", f"{h90:.2f}/{l90:.2f}")
    m4.metric("120D H/L", f"{h120:.2f}/{l120:.2f}")
    m5.metric("52W H/L", f"{h52:.2f}/{l52:.2f}")

    # ADR Display
    st.code(f"ADR 10D: {adr_val:.2f} | ADR Upper: {curr_p+adr_val:.2f} | ADR Lower: {curr_p-adr_val:.2f}")

    # ===============================
    # THE TABS
    # ===============================
    t_close, t_demand, t_supply, t_vol, t_profile, t_readme = st.tabs([
        "🎯 Close", "📉 Low", "📈 High", "⚖️ Mid", "📊 Volume Profile", "📖 Guide"
    ])

    with t_profile:
        # Profile Logic
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig = go.Figure(go.Bar(x=counts, y=bin_centers, orientation='h', marker_color='rgba(255, 223, 128, 0.4)', name="Profile"))
        
        # Dashline Price Levels
        levels = [
            (poc_mid, "#FF0000", "POC", "dot"),
            (curr_p, "#00FF00", "NOW", "solid"),
            (vwap, "white", "VWAP", "dash"),
            (h90, "cyan", "90DH", "dashdot"), (l90, "cyan", "90DL", "dashdot"),
            (h120, "steelblue", "120DH", "dashdot"), (l120, "steelblue", "120DL", "dashdot"),
            (h52, "gray", "52WH", "dashdot"), (l52, "gray", "52WL", "dashdot")
        ]

        for val, col, label, dash in levels:
            fig.add_shape(type="line", x0=0, x1=max(counts), y0=val, y1=val, line=dict(color=col, width=2, dash=dash))
            fig.add_annotation(x=max(counts), y=val, text=label, showarrow=False, xanchor="left", font=dict(color=col, size=12))

        fig.update_layout(template="plotly_dark", height=700, margin=dict(l=20, r=80, t=20, b=20), xaxis_visible=False, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with t_close:
        display_full_table(calculate_zones(df, 'Close', tol), curr_p, "1. Confirmation (CLOSE)", "Confirmation", adr=adr_val, last_time=last_time)

    with t_demand:
        display_full_table(calculate_zones(df, 'Low', tol), curr_p, "2. Demand (LOW)", "Support", adr=adr_val, last_time=last_time)

    with t_supply:
        display_full_table(calculate_zones(df, 'High', tol), curr_p, "3. Supply (HIGH)", "Resistance", adr=adr_val, last_time=last_time)

    with t_vol:
        df['Mid_Price'] = (df['High'] + df['Low']) / 2
        display_full_table(calculate_zones(df, 'Mid_Price', tol), curr_p, "4. Volatility (MID)", "Pivot", adr=adr_val, last_time=last_time)

    with t_readme:
        st.markdown("""
        ### 🚀 How to Trade with Price Zones
        
        This tool identifies high-density price clusters using historical daily data. It helps find where 'Big Money' has historically supported or resisted the price.

        ---
        #### 1. Finding Your Entry (Demand/Low)
        * **Switch to the 📉 Low Tab:** This shows clusters of historical daily lows (Support).
        * **Look for 'Very Strong' Touch-Rank:** This indicates a price level that has been tested many times.
        * **Entry Signal:** If the current price is approaching a **Very Strong** zone from above, look for a bounce.
        * **Manipulation Check:** If `Manip?` is **YES**, it means that zone has extreme volume per touch. This is often where institutions are "cleaning" orders.

        #### 2. Finding Your Exit (Supply/High)
        * **Switch to the 📈 High Tab:** This shows clusters of historical daily highs (Resistance).
        * **Targeting:** Look for the nearest 'Strong' or 'Very Strong' zone above your entry price.
        * **The Pink Row:** In the tables, the row highlighted in **Magenta/Pink** is the zone closest to the current market price.

        #### 3. Using the Volume Profile
        

[Image of volume profile chart with point of control]

        * **POC (Red Dot):** The Point of Control. Price is naturally "attracted" to this level. If price is far above POC, it might be overextended.
        * **VWAP (White Dash):** The average price weighted by volume. Staying above VWAP is generally bullish.

        #### 4. The "Confirmation" (Close) Tab
        * Use this tab to see where the stock actually *finishes* the day. If a stock is trading at $105, but there is a **Very Strong Close Zone** at $102, the stock has a high probability of "settling" back at $102 by end of day.

        #### 🎯 Summary Strategy:
        1.  **Identify Support:** Use the **Low** tab to find a "Very Strong" zone below price.
        2.  **Verify Volume:** Ensure the **Trade-Rank** is also Medium or Strong.
        3.  **Set Target:** Use the **High** tab to find the next "Strong" resistance zone.
        4.  **Risk Management:** Your stop-loss should generally be just below the **Zone Low** of your entry cluster.
        """, unsafe_allow_html=True)
