import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ===============================
# CONFIG
# ===============================
PERCENT_STEP = 0.008   # 0.8%
MIN_PCT = 0.50
MAX_PCT = 1.50

st.set_page_config(layout="wide", page_title="ETF Calculator")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 22px !important; margin-bottom: 10px !important; }
        .text-table {
            font-family: 'Courier New', monospace; white-space: pre;
            background-color: #0E1117; padding: 15px;
            border: 1px solid #31333F; border-radius: 8px;
            font-size: 16px; line-height: 1.4; width: 100%;
        }
        .guide-box {
            background-color: #262730; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌅 ETF Leveraged Price Calculator</h1>", unsafe_allow_html=True)

# Create Tabs
tab1, tab2 = st.tabs(["📊 Price Calculator", "📖 Read Me / Guide"])

with tab1:
    # =====================================================
    # SMART PATH CONFIGURATION (LOCAL & GITHUB)
    # =====================================================
    LOCAL_PATH = r"C:\Users\swong\Owner\PYTHON\US Trading\Ticker\Leveraged Mapping.xlsx"
    GITHUB_PATH = os.path.join("log files", "Leveraged Mapping.xlsx")

    if os.path.exists(LOCAL_PATH):
        FINAL_LOG_PATH = LOCAL_PATH
    elif os.path.exists(GITHUB_PATH):
        FINAL_LOG_PATH = GITHUB_PATH
    else:
        FINAL_LOG_PATH = "Leveraged Mapping.xlsx"

    # ===============================
    # LOAD ETF MAPPING
    # ===============================
    etf_mapping = {}

    try:
        df_map = pd.read_excel(FINAL_LOG_PATH, usecols=["Ticker", "Type", "ETF", "Leverage"])
        for _, row in df_map.iterrows():
            underlying = str(row["Ticker"]).strip().upper()
            direction = str(row["Type"]).strip().upper()
            ticker_val = str(row["ETF"]).strip().upper()
            lev = float(row["Leverage"])

            if underlying not in etf_mapping:
                etf_mapping[underlying] = {"bull": [], "bear": []}

            if direction == "BULL":
                etf_mapping[underlying]["bull"].append({"ticker": ticker_val, "leverage": lev})
            elif direction == "BEAR":
                etf_mapping[underlying]["bear"].append({"ticker": ticker_val, "leverage": lev})
    except Exception as e:
        st.error(f"Error loading mapping file: {e}")
        st.stop()

    # ===============================
    # INPUT
    # ===============================
    st.markdown("<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>Enter Underlying Ticker (e.g. SOXX, QQQ, SPY)</p>", unsafe_allow_html=True)
    col_input, col_btn, _ = st.columns([2, 1, 3])
    with col_input:   
        ticker_input = st.text_input("Ticker", value="SOXX", label_visibility="collapsed").strip().upper()
    with col_btn:
        if st.button("🔄"):
            st.rerun()

    if ticker_input not in etf_mapping:
        st.warning(f"No ETF mapping found for '{ticker_input}'. Check your Excel file.")
    else:
        # ===============================
        # PRICE FETCH & CALCULATION
        # ===============================
        def fetch_price(t):
            try:
                data = yf.download(t, period="1d", interval="1m", prepost=True, progress=False, auto_adjust=True)
                if not data.empty:
                    return round(float(data["Close"].iloc[-1]), 2)
                hist = yf.Ticker(t).history(period="1d")
                return round(float(hist["Close"].iloc[-1]), 2) if not hist.empty else None
            except:
                return None

        with st.spinner("Fetching market prices..."):
            stock_price = fetch_price(ticker_input)
            if stock_price is None:
                st.error(f"Unable to fetch price for {ticker_input}")
            else:
                bull_prices = [(e["ticker"], fetch_price(e["ticker"]), e["leverage"]) for e in etf_mapping[ticker_input]["bull"]]
                bear_prices = [(e["ticker"], fetch_price(e["ticker"]), e["leverage"]) for e in etf_mapping[ticker_input]["bear"]]

                # Generate Table Data
                data_rows = []
                pct_values = np.arange(MAX_PCT, MIN_PCT - (PERCENT_STEP/2), -PERCENT_STEP)

                for pct in pct_values:
                    row = {"Pct%": f"{round(pct * 100, 2)}%",
                           ticker_input: round(stock_price * pct, 2)}

                    for t, p, lev in bull_prices:
                        if p: row[f"{t}({lev}x)"] = round(p * (1 + lev*(pct - 1)), 2)
                    for t, p, lev in bear_prices:
                        if p: row[f"{t}({lev}x)"] = round(p * (1 - lev*(pct - 1)), 2)
                    
                    data_rows.append(row)

                df_display = pd.DataFrame(data_rows)

                # ===============================
                # DISPLAY
                # ===============================
                st.code(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')} | {ticker_input} Base Price: {stock_price}")

                header = "".join([f"{col:>15}" for col in df_display.columns])
                separator = "-" * (15 * len(df_display.columns))

                rows_html = ""
                for _, row in df_display.iterrows():
                    line = "".join([f"{str(val):>15}" for val in row])
                    rows_html += line + "\n"

                st.markdown(f"""
                    <div class="text-table">
{header}
{separator}
{rows_html}    </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    ### How to Use This Tool
    This calculator helps you visualize the **theoretical price** of leveraged ETFs (Bull and Bear) based on the price movement of their underlying index.
    
    #### 1. The Core Concept
    Leveraged ETFs are designed to multiply the **daily** performance of an index. 
    * **Bull (3x):** If the index goes up 1%, the ETF should go up ~3%.
    * **Bear (3x):** If the index goes down 1%, the ETF should go up ~3%.
    
    #### 2. How to Read the Table
    Go to the **Price Calculator** tab and enter a ticker (e.g., `SOXX`). The table will generate a range of price targets:
    
    * **Pct% Column:** This represents the target price of the underlying index relative to its current price (100% = No change).
    * **Ticker Column:** The projected price of the base index.
    * **Leveraged Columns (e.g., SOXL 3x):** The calculated price the ETF *should* reach if the index hits that specific target.
    
    #### 3. Pro-Tips for Checking Correspondence
    * **Support/Resistance:** Find a major support level for the underlying index (e.g., SPY at $500). Look up $500 in the table to see what the corresponding price for SPXL (Bull) or SPXS (Bear) would be.
    * **Volatility Decay:** Remember that these calculations are based on the **current** spot price. Because of "Beta Slippage," leveraged ETFs are best for short-term targets.
    * **Refresh:** Use the 🔄 button to get the most recent market prices before making a calculation.
    """)
    
    st.info("💡 **Note:** Calculations use the formula: $Price_{new} = Price_{current} \\times (1 + (Leverage \\times \\Delta Index))$")
