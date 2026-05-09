import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# ============================================
# CONFIG & PATHS
# ============================================
LOOKBACK_DAYS = 10 
BARS_PER_DAY_30M = 32  
BARS_PER_DAY_1H = 16

BASE_DIR = r"C:\Users\swong\Owner\Python\US Trading\Ticker"
TICKER_FOLDER = BASE_DIR if os.path.exists(BASE_DIR) else "ticker"

# --- UI CONFIG ---
st.set_page_config(layout="wide", page_title="Filter ABC")
st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        html, body, [class*="css"] { font-size: 14px !important; }
        h1 { font-size: 18px !important; margin-bottom: 10px !important; }
        div[data-testid="stDataFrame"] td { font-size: 12px !important; }
        [data-testid="stDataFrame"] div[data-testid="stTable"] td { padding: 2px 5px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>📊 Filter ABC</h1>", unsafe_allow_html=True)

if "results_30m" not in st.session_state: st.session_state.results_30m = pd.DataFrame()
if "results_1h" not in st.session_state: st.session_state.results_1h = pd.DataFrame()

# --- DATA ENGINE ---
def get_processed_data(ticker, interval):
    window = LOOKBACK_DAYS * (BARS_PER_DAY_30M if interval == "30m" else BARS_PER_DAY_1H)
    df_math = yf.download(ticker, period="20d", interval=interval, prepost=True, auto_adjust=True, progress=False)
    
    if df_math.empty: return None
    if isinstance(df_math.columns, pd.MultiIndex): 
        df_math.columns = df_math.columns.get_level_values(0)

    # WVAD Calculation
    hl_range = (df_math['High'] - df_math['Low']).replace(0, 0.0001)
    raw_wvad = ((df_math['Close'] - df_math['Open']) / hl_range) * df_math['Volume']
    df_math['WVAD'] = raw_wvad.rolling(window=window).sum()

    # CCI Calculation
    tp = (df_math['High'] + df_math['Low'] + df_math['Close']) / 3
    sma = tp.rolling(window=window).mean()
    mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df_math['CCI'] = (tp - sma) / (0.015 * mad)

    return df_math.tail(window).astype(float)

def find_recursive_chain(df):
    close, times = df['Close'].values, df.index
    patterns, cursor = [], 2
    while cursor < len(close) - 1:
        A, B, C = None, None, None
        for i in range(cursor, len(close) - 1):
            if close[i] < close[i-1] and close[i] < close[i+1]:
                A = {'time': times[i], 'price': close[i], 'idx': i}; break
        if not A: break
        found_reset = False
        for j in range(A['idx'] + 1, len(close)):
            if close[j] < A['price']: cursor, found_reset = j, True; break
            if j < len(close) - 1 and close[j] > close[j-1] and close[j] > close[j+1]:
                B = {'time': times[j], 'price': close[j], 'idx': j}; break
        if found_reset: continue
        if not B: patterns.append({'A': A, 'B': None, 'C': None, 'state': 'SEARCHING_FOR_B'}); break
        for k in range(B['idx'] + 1, len(close)):
            if close[k] < A['price']: cursor, found_reset = k, True; break
            if k < len(close) - 1 and close[k] < close[k-1] and close[k] < close[k+1]:
                if close[k] > A['price']:
                    C = {'time': times[k], 'price': close[k], 'idx': k}
                    patterns.append({'A': A, 'B': B, 'C': C, 'state': 'CONFIRMED_C'})
                    cursor = k + 1; break
                else: cursor, found_reset = k, True; break
        if found_reset: continue
        if not C: patterns.append({'A': A, 'B': B, 'C': None, 'state': 'WAITING_FOR_C'}); break
    return patterns

def scan_tickers(ticker_list, interval):
    results = []
    progress_bar = st.progress(0)
    for i, tkr in enumerate(ticker_list):
        try:
            tkr_clean = str(tkr).strip().upper()
            df = get_processed_data(tkr_clean, interval)
            if df is None: continue
            
            df_5m = yf.download(tkr_clean, period="1d", interval="5m", prepost=True, auto_adjust=True, progress=False)
            if df_5m.empty: continue
            if isinstance(df_5m.columns, pd.MultiIndex): df_5m.columns = df_5m.columns.get_level_values(0)
            
            live_p = df_5m['Close'].iloc[-1]
            vwap = (df_5m['Close'] * df_5m['Volume']).cumsum() / df_5m['Volume'].cumsum()
            last_vwap = vwap.iloc[-1]
            
            patterns = find_recursive_chain(df)
            wvad_now = df["WVAD"].iloc[-1]
            cci_now = df["CCI"].iloc[-1]

            p = patterns[-1] if patterns else {'state': 'NO_PATTERN', 'A': {'price': 0}, 'B': None, 'C': None}
            a_p = p['A']['price']
            b_p = p['B']['price'] if p.get('B') else 0
            c_p = p['C']['price'] if p.get('C') else 0
            
            # --- RENAMED FLAG LOGIC ---
            state_map = {
                "CONFIRMED_C": "CONSIDER ENTER" if wvad_now > 0 else "C (NO VOLUME)", 
                "WAITING_FOR_C": "Wait C > A"
            }
            flag = state_map.get(p['state'], "NO PATTERN")
            
            scen = "🟢 CONFIRMED C + ACCUMULATION" if flag == "CONSIDER ENTER" \
                   else "⚪ C (LOW VOLUME)" if flag == "C (NO VOLUME)" \
                   else "⏳ Wait C > A" if flag == "Wait C > A" \
                   else "🔍 NO PATTERN"
                   
            results.append({
                "Ticker": tkr_clean, "FLAG": flag, "Scenario": scen, 
                "WVAD (10DAY)": wvad_now, "CCI (10DAY)": cci_now,
                "Price Now": live_p, "VWAP (5MIN)": last_vwap, 
                "Point A": a_p, 
                "Point B": b_p if b_p > 0 else np.nan, 
                "Point C": c_p if c_p > 0 else np.nan
            })
        except: continue
        progress_bar.progress((i + 1) / len(ticker_list))
    progress_bar.empty()
    cols = ["Ticker", "FLAG", "Scenario", "WVAD (10DAY)", "CCI (10DAY)", "Price Now", "VWAP (5MIN)", "Point A", "Point B", "Point C"]
    return pd.DataFrame(results, columns=cols)

# --- READABLE STYLING ---
def format_readable(df):
    if df.empty: return df
    
    prio = {"CONSIDER ENTER": 0, "C (NO VOLUME)": 1, "Wait C > A": 2, "NO PATTERN": 3}
    df = df.assign(sort=df['FLAG'].map(prio).fillna(4)).sort_values('sort').drop('sort', axis=1)

    def human_format(num):
        if not isinstance(num, (int, float)): return num
        mag, abs_n = 0, abs(num)
        while abs_n >= 1000: mag += 1; abs_n /= 1000.0
        return '%.2f%s' % (abs_n * (1 if num >= 0 else -1), ['', 'K', 'M', 'G', 'T'][mag])

    def price_fmt(x):
        # pd.isna checks for np.nan values
        if pd.isna(x) or x == 0: return "-"
        return "{:,.2f}".format(x) if isinstance(x, (int, float)) else x

    return df.style.map(
        lambda x: 'color:#00ff00;' if isinstance(x,(int,float)) and x > 0 else 'color:#ff4b4b;', 
        subset=['WVAD (10DAY)', 'CCI (10DAY)']
    ).format({
        "WVAD (10DAY)": human_format, 
        "CCI (10DAY)": "{:.2f}",
        "Price Now": price_fmt,
        "VWAP (5MIN)": price_fmt,
        "Point A": price_fmt,
        "Point B": price_fmt,
        "Point C": price_fmt
    })

# --- EXECUTION ---
NY_TZ = ZoneInfo("America/New_York")
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

col_input, col_refresh = st.columns([1, 1])

with col_input:
    st.markdown(
        f"<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>"
        f"Enter CSV Name  |  Last refresh (NY) : "
        f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True
    )
    file_input = st.text_input("CSV Name", value="WATCH", label_visibility="collapsed").strip().upper()

with col_refresh:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("🔄"):
        st.session_state.results_30m = pd.DataFrame()
        st.session_state.results_1h = pd.DataFrame()
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.cache_data.clear()
        st.rerun()
        
csv_path = os.path.join(TICKER_FOLDER, f"{file_input}.csv")

if os.path.exists(csv_path):
    tkrs = pd.read_csv(csv_path).iloc[:, 0].dropna().tolist()
    tabs = st.tabs(["📊 30MINUTES SCAN", "📊 ONE HOUR SCAN"])
    for tab, interval in zip(tabs, ["30m", "1h"]):
        with tab:
            if st.button(f"🚀 Run {interval.upper()} Scan"):
                st.session_state[f"results_{interval}"] = scan_tickers(tkrs, interval)
            
            res_df = st.session_state.get(f"results_{interval}", pd.DataFrame())
            if not res_df.empty:
                st.dataframe(format_readable(res_df), width="stretch", hide_index=True)
else:
    st.error(f"File not found: {csv_path}")
