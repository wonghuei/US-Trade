import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pytz

# ===============================
# CONFIG & CSS
# ===============================
st.set_page_config(layout="wide", page_title="Price Zone")

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

st.markdown("<h1>Price Zone + Volume Profile</h1>", unsafe_allow_html=True)

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
    
    prices = df[price_column].values
    volumes = df['Volume'].values
    dates = df.index
    today = dates[-1]

    zones = []
    for p, v, d in zip(prices, volumes, dates):
        # Logic: 2.0 for < 30 days, 1.5 for < 90 days, 1.0 otherwise
        days_ago = (today - d).days
        weight = 2.0 if days_ago <= 30 else 1.5 if days_ago <= 90 else 1.0
        
        found = False
        for z in zones:
            if abs(p - z['Zone Mid']) / z['Zone Mid'] <= tolerance:
                z['Touches'] += 1
                z['Weighted_Score'] += weight  # Hidden power score
                z['Recent_Hits'] += 1 if days_ago <= 30 else 0 # Explicit counter
                z['Prices'].append(float(p))
                z['Total Vol'] += float(v)
                z['Zone Mid'] = float(np.median(z['Prices']))
                found = True; break
        if not found:
            zones.append({
                'Zone Mid': float(p), 
                'Touches': 1, 
                'Weighted_Score': weight,
                'Recent_Hits': 1 if days_ago <= 30 else 0,
                'Prices': [float(p)], 
                'Total Vol': float(v)
            })
    
    z_df = pd.DataFrame(zones)
    if z_df.empty: return z_df
    
    z_df = z_df[z_df['Touches'] >= MIN_TOUCHES].copy()
    
    # Calculate key Flag
    # If more than 30% of touches are recent, we flag it as "HOT"
    z_df['Key'] = (z_df['Recent_Hits'] / z_df['Touches']).apply(
        lambda x: "🔥" if x > 0.4 else "⚡" if x > 0.1 else "🏛"
    )

    # Ranking based on Weighted Score (The Key Bias)
    max_w = z_df['Weighted_Score'].max()
    z_df['Touch-Rank'] = (z_df['Weighted_Score'] / max_w).apply(
        lambda s: "Very Strong" if s >= 0.8 else "Strong" if s >= 0.6 else "Medium" if s >= 0.4 else "Weak")
    
    # ... (rest of your existing Low/High/Vol calculations)
    z_df['Zone Low'] = z_df['Prices'].apply(min)
    z_df['Zone High'] = z_df['Prices'].apply(max)
    z_df['Vol/Touch'] = z_df['Total Vol'] / z_df['Touches']
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
    html_output += "<span>" + "-" * 160 + "</span>\n"
    
    # CORRECTED HEADERS: Just use the word 'Key' as a label
    headers = (f"{'Zone Low':>11} {'Zone High':>11} {'Zone Mid':>11} {'Touch':>8} {'Key':>9} "
               f"{'Touch-Rank':>13} {'Touch-Vol':>18} {'Diff($)':>9} {'Diff(%)':>9} "
               f"{'Trade-Rank':>13} {'Trade-Vol':>15} {'Manip?':>8}\n")
    
    html_output += f"<span>{headers}</span><span>" + "-" * 160 + "</span>\n"

    for idx, row in zones_df.iterrows():
        diff_dollar = row['Zone Mid'] - current_price
        diff_pct = (diff_dollar / current_price) * 100
        
        # Color Logic
        color = "#FFFFFF"
        if row['Touch-Rank'] == "Very Strong": color = "#00FF00"
        elif row['Touch-Rank'] == "Strong": color = "#FFFF00"
        elif row['Touch-Rank'] == "Medium": color = "#00FFFF"
        
        # Highlight current price area
        if idx == closest_idx: color = "#FF00FF"
        
        # Glow effect for HOT zones
        row_style = f'color:{color};'
        if "🔥" in str(row['Key']):
            row_style += "font-weight:bold; background-color: rgba(255, 69, 0, 0.1);"

        manip = "YES" if row['Manip?'] else ""
        
        # CORRECTED LINE: This is where we actually use row['Key']
        line = (f"{row['Zone Low']:>11.2f} {row['Zone High']:>11.2f} {row['Zone Mid']:>11.2f} "
                f"{row['Touches']:>8} {row['Key']:>9} {row['Touch-Rank']:>13} {row['Vol/Touch']:>18,.0f} "
                f"{diff_dollar:>9.2f} {diff_pct:>9.2f}% "
                f"{row['Trade-Rank']:>13} {row['Total Vol']:>15,.0f} {manip:>8}\n")
        
        html_output += f'<span style="{row_style}">{line}</span>'
        
    st.markdown(html_output + "</div>", unsafe_allow_html=True)

# ===============================
# UI INPUTS
# ===============================
import streamlit as st

col1, col2, col_btn = st.columns([1, 2, 1]) # Adjusted ratios for better fit

with col1:
    ticker = st.text_input("Enter Ticker", value="SOXX", autocomplete="off").upper()

with col2:
    lookback = st.radio("Historical Range", ["1y", "2y", "3mo", "6mo"], index=0, horizontal=True)

with col_btn:
    # Add vertical space to align with the labels of the other widgets
    st.markdown('<p style="margin-bottom: 24px;"></p>', unsafe_allow_html=True) 
    run_scan = st.button("🚀 Run Scan", use_container_width=True)
    
if ticker:
    # 1. Get Hourly data to calculate the "Synthetic Close"
    # Note: 2y is the max for 1h data on yfinance
    df_h = yf.download(ticker, period="2y", interval="1h", progress=False)
    
    # 2. Get 5m data for the "Live" price
    df_now = yf.download(ticker, period="1d", interval="5m", prepost=True, progress=False)
    
    if df_h.empty or df_now.empty: 
        st.warning("Data not found. Check the ticker or try '2y' lookback.")
        st.stop()

    # Clean multi-index columns if they exist
    if isinstance(df_h.columns, pd.MultiIndex): df_h.columns = df_h.columns.get_level_values(0)
    if isinstance(df_now.columns, pd.MultiIndex): df_now.columns = df_now.columns.get_level_values(0)

    # 3. Calculate your Synthetic Price per day
    def calc_synthetic(day_group):
        if len(day_group) >= 8:
            # Taking the Close price at the end of hours 1, 2, 4, and 8
            h1 = day_group['Close'].iloc[0]
            h2 = day_group['Close'].iloc[1]
            h4 = day_group['Close'].iloc[3]
            h8 = day_group['Close'].iloc[7]
            return (h1 + h2 + h4 + h8) / 4
        return day_group['Close'].iloc[-1] # Fallback for half-days (holidays)

    # Create a daily dataframe by resampling the hourly data
    df = df_h.resample('D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Map the synthetic average back to the daily dataframe
    synthetic_results = df_h.groupby(df_h.index.date).apply(calc_synthetic)
    df['Synthetic_Close'] = synthetic_results

    # Core Calculations
    curr_p = float(df_now['Close'].iloc[-1])
    last_time = df_now.index[-1].strftime('%H:%M:%S')
    adr_val = float((df['High'].tail(10) - df['Low'].tail(10)).mean())
    
    # 1. ANCHOR VWAP (The cumulative average for the entire lookback period)
    # This represents the "Long Term Value"
    anchor_vwap = (df['Close'] * df['Volume']).sum() / df['Volume'].sum()
    
    # 2. DAILY VWAP (The weighted average of just the most recent trading session)
    # Note: For accuracy on the daily, we use the High/Low/Close of the current day
    last_day = df.iloc[-1]
    daily_vwap = (last_day['High'] + last_day['Low'] + last_day['Close']) / 3
    
    # --- UPGRADED KAMA LOGIC ---
    # 20-period for the overall Trend (Slow)
    df["KAMA_Trend"] = kama_dynamic(df["Close"], lookback=20)
    
    # 5-period for the Daily momentum (Fast)
    df["KAMA_Daily"] = kama_dynamic(df["Close"], lookback=5)
    
    # Get current values for the Metric display
    curr_kama_t = df["KAMA_Trend"].iloc[-1]
    curr_kama_d = df["KAMA_Daily"].iloc[-1]
    
    # Calculate the Tolerance for your Zones
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
    st.markdown('<p style="font-size:15px; color: yellow; margin-bottom:2px;">Price Updates as Per Trend or Per Average Period', unsafe_allow_html=True)
    curr_kama_d = df['KAMA_Daily'].iloc[-1]
    curr_kama_t = df['KAMA_Trend'].iloc[-1]
    r1, r2, r3, r4, r5 = st.columns(5, gap="small")
    r1.metric("PRICE NOW", f"{curr_p:.2f}")
    r2.metric("DLY VWAP/KAMA", f"{daily_vwap:.2f} / {curr_kama_d:.2f}", f"{daily_vwap - curr_kama_d:.2f}")
    r3.metric("VWAP (ANCHOR)", f"{anchor_vwap:.2f}")
    r4.metric("KAMA TREND", f"{curr_kama_t:.2f}", f"{curr_kama_d - curr_kama_t:.2f}")
    
    # Column 5: Volume Center
    r5.metric("POC Range", f"{poc_low:.2f} - {poc_high:.2f}", f"Mid: {poc_mid:.2f}")
    m1, m2, m3, m4, m5 = st.columns(5, gap="small")
    m1.metric("30D H/L", f"{h30:.2f}/{l30:.2f}")
    m2.metric("60D H/L", f"{h60:.2f}/{l60:.2f}")
    m3.metric("90D H/L", f"{h90:.2f}/{l90:.2f}")
    m4.metric("120D H/L", f"{h120:.2f}/{l120:.2f}")
    m5.metric("52W H/L", f"{h52:.2f}/{l52:.2f}")

    # ADR Display
    st.code(f"ADR 10D: {adr_val:.2f} | ADR Upper: {curr_p+adr_val:.2f} | ADR Lower: {curr_p-adr_val:.2f}")
    #st.markdown('<p style="font-size:12px; color:white; margin-bottom:2px;">🔥 (HOT) | ⚡ (NEW) | 🏛 (HIST), unsafe_allow_html=True)

    # ===============================
    # THE TABS
    # ===============================
    # 1. Place the Legend OUTSIDE and ABOVE the tabs
    st.markdown("""
    <div style="background-color: #1E2329; padding: 10px; border-radius: 5px; border-left: 5px solid #FF4500; margin-bottom: 15px;">
        <span style="font-size: 13px; color: white; font-weight: bold;">ZONE MATURITY KEY:</span>
        <span style="font-size: 12px; color: #CCCCCC; margin-left: 15px;">🔥 <b>HOT:</b> High activity in last 30 days (Active)</span>
        <span style="font-size: 12px; color: #CCCCCC; margin-left: 15px;">⚡ <b>NEW:</b> Significant hits in last 90 days (Waking Up)</span>
        <span style="font-size: 12px; color: #CCCCCC; margin-left: 15px;">🏛️ <b>HIST:</b> Older historical levels (Legacy Support/Res)</span>
    </div>
    """, unsafe_allow_html=True)

    # 2. Define the Tabs
    t_close, t_demand, t_supply, t_vol, t_profile = st.tabs([
        "🎯 Close", "📉 Low", "📈 High", "⚖️ Mid", "📊 Volume Profile"
    ])

    with t_close:
        display_full_table(calculate_zones(df, 'Close', tol), curr_p, "1. Confirmation (CLOSE)", "Confirmation", adr=adr_val, last_time=last_time)

    with t_demand:
        display_full_table(calculate_zones(df, 'Low', tol), curr_p, "2. Demand (LOW)", "Support", adr=adr_val, last_time=last_time)

    with t_supply:
        display_full_table(calculate_zones(df, 'High', tol), curr_p, "3. Supply (HIGH)", "Resistance", adr=adr_val, last_time=last_time)

    with t_vol:
        df['Mid_Price'] = (df['High'] + df['Low']) / 2
        display_full_table(calculate_zones(df, 'Mid_Price', tol), curr_p, "4. Volatility (MID)", "Pivot", adr=adr_val, last_time=last_time)

    with t_profile:
        # 1. High Resolution (150 bins) for sharp spikes
        counts, bin_edges = np.histogram(df['Close'].values, bins=100, weights=df['Volume'].values)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig = go.Figure(go.Bar(
            x=counts, 
            y=bin_centers, 
            orientation='h', 
            marker=dict(
                color='rgba(255, 255, 0, 0.5)',     # Bright Yellow (semi-transparent)
                line=dict(
                    color='rgba(255, 255, 255, 0.8)', # Sharp White Outline
                    width=0.5                        # Thin enough to not be "chunky"
                )
            ),
            name="Volume Profile"
        ))
        
        # Dashline Price Levels
        levels = [
            (poc_mid, "#FF0000", "POC", "dot"),
            (curr_p, "#00FF00", "NOW", "solid"),
            (daily_vwap, "pink", "VWAP(DAILY)", "solid"),
            (anchor_vwap, "white", "VWAP(ANCHOR", "dash"),
            (curr_kama_t, "#FFA500", "KAMA-Trend", "dash"), # Orange
            (curr_kama_d, "#FF8C00", "KAMA-Daily", "solid"), # Dark Orange
            (h30, "blue", "30DH", "dashdot"), (l30, "blue", "30DL", "dashdot"),
            (h60, "brown", "60DH", "dashdot"), (l60, "brown", "60DL", "dashdot"),
            (h90, "cyan", "90DH", "dashdot"), (l90, "cyan", "90DL", "dashdot"),
            (h120, "steelblue", "120DH", "dashdot"), (l120, "steelblue", "120DL", "dashdot"),
            (h52, "gray", "52WH", "dashdot"), (l52, "gray", "52WL", "dashdot")
        ]
        
        for val, col, label, dash in levels:
            fig.add_shape(type="line", x0=0, x1=max(counts), y0=val, y1=val, line=dict(color=col, width=2, dash=dash))
            fig.add_annotation(x=max(counts), y=val, text=label, showarrow=False, xanchor="left", font=dict(color=col, size=12))

        fig.update_layout(
            template="plotly_dark", 
            height=1000, 
            bargap=0,        # Bars touch
            bargroupgap=0,   # Groups touch
            margin=dict(l=20, r=100, t=20, b=20), 
            xaxis_visible=False, 
            yaxis=dict(
                showgrid=False, 
                zeroline=False,
                # This helps prevent the "fat bar" stretching
                nticks=20 
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
