import streamlit as st
import pandas as pd
import yfinance as yf
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# CONFIG & PATHS
# =====================================================
LOG_PATH = r"C:\Users\swong\Owner\Python\US Trading\Rpt Log\Log - Options Range.csv"
st.set_page_config(layout="wide", page_title="Trend Options Range")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>📈 Trend Option Range (Live)</h1>", unsafe_allow_html=True)

# =====================================================
# CORE ENGINES
# =====================================================
def mid_price(row):
    bid, ask, last = float(row.get("bid", 0)), float(row.get("ask", 0)), float(row.get("lastPrice", 0))
    return (bid + ask) / 2 if bid > 0 and ask > 0 else last

def get_live_market_context(ticker):
    try:
        t = yf.Ticker(ticker)
        # Fetch 1m interval for tightest spot price sync
        hist = t.history(period="1d", interval="1m", prepost=True)
        if hist.empty: return None
        spot = float(hist["Close"].iloc[-1])

        expirations = t.options
        if not expirations: return None
        chain = t.option_chain(expirations[0])
        
        calls, puts = chain.calls.copy(), chain.puts.copy()
        calls["diff"] = (calls["strike"] - spot).abs()
        atm_idx = calls["diff"].idxmin()
        
        strike = float(calls.loc[atm_idx, "strike"])
        c_mid = mid_price(calls.loc[atm_idx])
        
        matching_puts = puts[puts["strike"] == strike]
        if matching_puts.empty: return None
        p_mid = mid_price(matching_puts.iloc[0])

        return {
            "Spot": spot,
            "Live_ATM_Strike": strike,
            "Live_Call": c_mid,
            "Live_Put": p_mid,
            "Live_Straddle": c_mid + p_mid
        }
    except: return None

def load_and_sync_data(ticker):
    if not os.path.exists(LOG_PATH): return pd.DataFrame()

    df = pd.read_csv(LOG_PATH)
    df = df[df["Ticker"].str.upper() == ticker.upper()].copy()
    if df.empty: return pd.DataFrame()

    market = get_live_market_context(ticker)
    if not market: return pd.DataFrame()
    
    df["Period_End_Date"] = pd.to_datetime(df["Period_End_Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Period_End_Date"])
    
    spot = market["Spot"]
    strike = market["Live_ATM_Strike"]
    straddle = market["Live_Straddle"]
    
    # 1. Math Range (Symmetric)
    move = straddle - abs(strike - spot)
    move = max(move, 0)
    
    # 2. Skewed Levels (IBKR Model)
    c_time_val = max(market["Live_Call"] - max(spot - strike, 0), 0)
    p_time_val = max(market["Live_Put"] - max(strike - spot, 0), 0)

    # 3. Data Mapping
    df["Spot_Price"] = spot
    df["Price_Diff"] = spot - df["Stock_Price"]
    df["Diff_Pct"] = (df["Price_Diff"] / df["Stock_Price"]) * 100
    df["Lower_Range"] = spot - move
    df["Upper_Range"] = spot + move
    
    # CRITICAL: Adding the IBKR Columns here
    df["Skew Lower"] = spot - p_time_val
    df["Skew Upper"] = spot + c_time_val
    
    df["EM%"] = (move / spot) * 100
    df["Skew_Ratio"] = market["Live_Put"] / market["Live_Call"] if market["Live_Call"] > 0 else 1.0
    df["ATM_Call_Mid"] = market["Live_Call"]
    df["ATM_Put_Mid"] = market["Live_Put"]

    return df.sort_values("Period_End_Date")

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
    f"Enter Stock Ticker  |  Last refresh (AMS / NY) : "
    f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
    )
    ticker_input = st.text_input("Enter Ticker", value="APP", autocomplete="off", label_visibility="collapsed").strip().upper()
with c2:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("🔄 Calculator"):
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.rerun()

if ticker_input:
    df = load_and_sync_data(ticker_input)
    if not df.empty:
        tabs = st.tabs(["Coming Week", "Monthly", "Quarterly"])
        tf_mapping = ["Coming Week", "Monthly", "Quarterly"]

        for tab, tf in zip(tabs, tf_mapping):
            with tab:
                tf_df = df[df["Timeframe"] == tf].copy()
                if tf_df.empty:
                    st.info(f"No logged data for {tf}")
                    continue
                
                tf_df["Period_End_Date"] = tf_df["Period_End_Date"].dt.strftime('%d/%m/%Y')
                
                # --- THE DEFINITIVE COLUMN LIST ---
                show_cols = [
                    "Period_End_Date", 
                    "Spot_Price",
                    "ATM_Strike",
                    "Price_Diff", 
                    "Diff_Pct", 
                    "Lower_Range", 
                    "Upper_Range", 
                    "Skew Lower",       # Added
                    "Skew Upper",    # Added
                    "EM%", 
                    "Skew_Ratio", 
                    "ATM_Call_Mid", 
                    "ATM_Put_Mid"
                ]
                
                st.dataframe(
                    tf_df[show_cols].style.format({
                        "Spot_Price": "${:.2f}", 
                        "Price_Diff": "{:+.2f}",
                        "ATM_Strike": "${:.2f}", 
                        "Diff_Pct": "{:+.2f}%",
                        "Lower_Range": "${:.2f}", 
                        "Upper_Range": "${:.2f}",
                        "Skew Lower": "${:.2f}", 
                        "Skew Upper": "${:.2f}",
                        "EM%": "{:.2f}%", 
                        "Skew_Ratio": "{:.2f}",
                        "ATM_Call_Mid": "{:.2f}", 
                        "ATM_Put_Mid": "{:.2f}"
                    }), 
                    width="stretch", 
                    hide_index=True
                )
