import streamlit as st
import pandas as pd
import yfinance as yf
import math
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(layout="wide", page_title="Trend Options Range")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>📈 Trend Option Range (Live - All Expiries)</h1>", unsafe_allow_html=True)

# =====================================================
# CORE ENGINES
# =====================================================
def safe_float(x):
    try:
        if pd.isna(x): return 0.0
        return float(x)
    except: return 0.0

def get_mid_price(row):
    bid = safe_float(row.get('bid'))
    ask = safe_float(row.get('ask'))
    last = safe_float(row.get('lastPrice'))
    return (bid + ask) / 2 if bid > 0 and ask > 0 else last

def classify_skew(skew_ratio):
    if skew_ratio is None or skew_ratio == 0: return "Unknown"
    if skew_ratio < 0.8:  return "Upside Speculation"
    if skew_ratio <= 1.2: return "Neutral"
    if skew_ratio <= 1.5: return "Defensive"
    return "Extreme Put Skew"

def load_and_sync_data(ticker_symbol):
    t_obj = yf.Ticker(ticker_symbol)
    try:
        expirations = t_obj.options
        hist = t_obj.history(period="1d")
        if not expirations or hist.empty: return pd.DataFrame()
        
        price = float(hist['Close'].iloc[-1])
        today = datetime.now().date()
        results = []

        # Iterate through EVERY expiration available
        for exp in expirations:
            exp_date = pd.to_datetime(exp).date()
            dte = max((exp_date - today).days, 1)
            
            # Categorization logic from your original script
            if dte <= 14:
                timeframe = "Coming Week"
            elif dte <= 60:
                timeframe = "Monthly"
            elif dte <= 180:
                timeframe = "Quarterly"
            else:
                continue # Skip very long term to keep UI clean

            try:
                chain = t_obj.option_chain(exp)
                calls, puts = chain.calls.copy(), chain.puts.copy()
                if calls.empty or puts.empty: continue

                # Find ATM Strike
                calls['diff'] = (calls['strike'] - price).abs()
                puts['diff'] = (puts['strike'] - price).abs()
                atm_call = calls.sort_values('diff').iloc[0]
                atm_put = puts.sort_values('diff').iloc[0]

                # Calculations
                strike = safe_float(atm_call['strike'])
                c_mid = get_mid_price(atm_call)
                p_mid = get_mid_price(atm_put)
                
                straddle_move = c_mid + p_mid
                # Adjusted move based on your original logic
                straddle_adj = max(straddle_move - abs(strike - price), 0)

                # Calculate the ratio first
                skew_ratio = p_mid / c_mid if c_mid > 0 else 1.0

                results.append({
                    "Ticker": ticker_symbol,
                    "Timeframe": timeframe,
                    "Expiry": pd.to_datetime(exp),
                    "Spot_Price": price,
                    "ATM_Strike": strike,
                    "Price_Diff": price - strike, 
                    "Diff_Pct": ((price - strike) / strike) * 100 if strike > 0 else 0,
                    "Lower_Range": price - straddle_adj,
                    "Upper_Range": price + straddle_adj,
                    "Skew Lower": price - p_mid,
                    "Skew Upper": price + c_mid,
                    "EM%": (straddle_adj / price) * 100 if price > 0 else 0,
                    "Skew_Ratio": p_mid / c_mid if c_mid > 0 else 1.0,
                    "Status": classify_skew(skew_ratio),
                    "ATM_Call_Mid": c_mid,
                    "ATM_Put_Mid": p_mid
                })
            except:
                continue

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# =====================================================
# UI & DISPLAY
# =====================================================
NY_TZ = ZoneInfo("America/New_York")
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(
        f"<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>"
        f"Enter Stock Ticker  |  Last refresh (NY) : "
        f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True
    )
    ticker_input = st.text_input("Enter Ticker", value="APP", label_visibility="collapsed").strip().upper()
with c2:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Calculator"):
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.rerun()

if ticker_input:
    df = load_and_sync_data(ticker_input)
    if not df.empty:
        tabs = st.tabs(["Coming Week", "Monthly", "Quarterly"])
        tf_list = ["Coming Week", "Monthly", "Quarterly"]

        for tab, tf in zip(tabs, tf_list):
            with tab:
                # Filter for all dates belonging to this timeframe
                tf_df = df[df["Timeframe"] == tf].copy()
                
                if tf_df.empty:
                    st.info(f"No expirations found for {tf}")
                    continue
                
                tf_df["Expiry"] = tf_df["Expiry"].dt.strftime('%d/%m/%Y')
                
                show_cols = [
                    "Expiry", "Spot_Price", "ATM_Strike", "Price_Diff", 
                    "Diff_Pct", "Lower_Range", "Upper_Range", "Skew Lower", 
                    "Skew Upper", "EM%", "Skew_Ratio", "Status", "ATM_Call_Mid", "ATM_Put_Mid"
                ]
                
                st.dataframe(
                    tf_df[show_cols].style.format({
                        "Spot_Price": "${:.2f}", "Price_Diff": "{:+.2f}",
                        "ATM_Strike": "${:.2f}", "Diff_Pct": "{:+.2f}%",
                        "Lower_Range": "${:.2f}", "Upper_Range": "${:.2f}",
                        "Skew Lower": "${:.2f}", "Skew Upper": "${:.2f}",
                        "EM%": "{:.2f}%", "Skew_Ratio": "{:.2f}",
                        "ATM_Call_Mid": "{:.2f}", "ATM_Put_Mid": "{:.2f}"
                    }), 
                    width="stretch", hide_index=True, height=400
                )
    else:
        st.error("Could not fetch data. Check ticker or connection.")
