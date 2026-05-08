import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import os
from zoneinfo import ZoneInfo

# =====================================================
# 1. SMART PATH CONFIGURATION
# =====================================================
IS_LOCAL = os.path.exists(r"C:")
if IS_LOCAL:
    TICKER_DIR = r"C:\Users\swong\Owner\Python\US Trading\Ticker"
    OUTPUT_PATH = r"C:\Users\swong\Owner\Python\US Trading\Rpt Log\Log - Options Range.csv"
else:
    TICKER_DIR = "ticker" 
    CSV_LOG_PATH = os.path.join("log files", "Log - Options Range.csv")

st.set_page_config(layout="wide", page_title="Filter Options")

# UI STYLE
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
        [data-testid="stMetricValue"] { font-size: 15px !important; font-weight: 700; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 13px !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>📊 Filter Options Range (YF + Skew Analysis)</h1>", unsafe_allow_html=True)

# SAFE FUNCTIONS
def safe_float(x):
    try:
        return float(x) if pd.notna(x) else 0.0
    except:
        return 0.0

def mid_price(row):
    bid, ask, last = safe_float(row.get("bid")), safe_float(row.get("ask")), safe_float(row.get("lastPrice"))
    return (bid + ask) / 2 if bid > 0 and ask > 0 else last

# =====================================================
# CORE CALCULATION (MERGED LOGIC)
# =====================================================
def get_expected_range(ticker):
    t = yf.Ticker(ticker)
    try:
        # ALL DATA FROM YFINANCE (Spot Sync)
        hist = t.history(period="1d", interval="1m", prepost=True)
        if hist.empty: return None
        spot = float(hist["Close"].iloc[-1])

        expirations = t.options
        if not expirations: return None
        exp = expirations[0]
        chain = t.option_chain(exp)
        
        calls, puts = chain.calls.copy(), chain.puts.copy()
        calls["diff"] = (calls["strike"] - spot).abs()
        puts["diff"] = (puts["strike"] - spot).abs()
        atm_c = calls.sort_values("diff").iloc[0]
        atm_p = puts.sort_values("diff").iloc[0]

        strike = float(atm_c["strike"])
        c_mid, p_mid = mid_price(atm_c), mid_price(atm_p)
        
        # MATH SYNC (Your Original Move)
        move = (c_mid + p_mid) - abs(strike - spot)
        move = max(move, 0)

        # NEW IBKR ADAPTATIONS (Additional Columns)
        em_pct = (move / spot) * 100 if spot > 0 else 0
        skew_ratio = p_mid / c_mid if c_mid > 0 else 1.0
        
        # Skewed Support/Resistance (Time Value Logic)
        c_time_val = max(c_mid - max(spot - strike, 0), 0)
        p_time_val = max(p_mid - max(strike - spot, 0), 0)

        return {
            "Expiry": exp,
            "Ticker": ticker,
            "Spot": spot,
            "ATM Strike": strike,
            "Call Mid": c_mid,
            "Put Mid": p_mid,
            "Expected Move": move,
            "EM%": em_pct,               # NEW
            "Lower Range": spot - move,
            "Upper Range": spot + move,
            "Skew Ratio": skew_ratio,    # NEW
            "Skew Lower": spot - p_time_val,# NEW (Skewed Lower)
            "Skew Upper": spot + c_time_val # NEW (Skewed Upper)
        }
    except:
        return None

# CLASSIFICATION (Standardized IBKR 4-Tier Logic)
def classify(row):
    skew = row["Skew Ratio"]
    if skew < 0.8: 
        return "Upside Speculation"
    elif skew <= 1.2: 
        return "Neutral"
    elif skew <= 1.5: 
        return "Defensive"
    else: 
        return "Extreme Put Skew"

# COLOR LOGIC (Fixed Column Names)
def color_df(df):
    style = pd.DataFrame("", index=df.index, columns=df.columns)
    for i in df.index:
        spot, low, high = df.loc[i, "Spot"], df.loc[i, "Lower Range"], df.loc[i, "Upper Range"]
        if abs(spot - low) < abs(spot - high):
            style.loc[i, "Lower Range"] = "background-color: green; color: black; font-weight: bold"
        elif abs(spot - high) < abs(spot - low):
            style.loc[i, "Upper Range"] = "background-color: red; color: black; font-weight: bold"
    return style

# UI INPUTS
c1, c2, _ = st.columns([2, 1, 1], vertical_alignment="bottom")
NY_TZ = ZoneInfo("America/New_York")
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

with c1:
    st.markdown(f"<p style='color: #888; font-size: 12px;'>Last sync: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    file_input = st.text_input("Ticker File", value="TEST", label_visibility="collapsed").strip().upper()
    ticker_file = os.path.join(TICKER_DIR, f"{file_input}.csv")

with c2:
    run = st.button("🔄 Sync & Calculate")

# RUN LOGIC
if run:
    if not os.path.exists(ticker_file):
        st.error(f"File not found: {ticker_file}"); st.stop()

    tickers = pd.read_csv(ticker_file)["Ticker"].dropna().tolist()
    results = []
    prog = st.progress(0)
    for i, t in enumerate(tickers, 1):
        data = get_expected_range(str(t).strip().upper())
        if data: results.append(data)
        prog.progress(i / len(tickers))

    df = pd.DataFrame(results)
    if not df.empty:
        # INSERT STATUS
        df.insert(df.columns.get_loc("Upper Range") + 1, "Status", df.apply(classify, axis=1))

        # ROUNDING (Including New Columns)
        price_cols = ["ATM Strike", "Call Mid", "Put Mid", "Expected Move", "Spot", 
                      "Lower Range", "Upper Range", "Skew Lower", "Skew Upper", "Skew Ratio", "EM%"]
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').round(2)

        # METRICS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Tickers", len(df))
        m2.metric("🔵 Upside", (df["Status"] == "Upside Speculation").sum())
        m3.metric("⚪ Neutral", (df["Status"] == "Neutral").sum())

        # We group 'Defensive' and 'Extreme' into one 'Risk/Put Skew' metric
        put_skew_count = df["Status"].isin(["Defensive", "Extreme Put Skew"]).sum()
        m4.metric("🔴 Put Skew", put_skew_count)

        # DISPLAY
        st.dataframe(
            df.style.format({
                "EM%": "{:.2f}%", 
                "Skew Ratio": "{:.2f}",
                **{c: "{:.2f}" for c in price_cols if c not in ["EM%", "Skew Ratio"]}
            }).apply(color_df, axis=None), 
            width="stretch", 
            hide_index=True
        )
        st.session_state.last_refresh = datetime.now(NY_TZ)
