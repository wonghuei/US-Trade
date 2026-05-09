import streamlit as st
import pandas as pd
import yfinance as yf
import os
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# --- Page Config ---
st.set_page_config(page_title="Ribbon EMA", layout="wide")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 20px !important; margin-bottom: 0px !important; }
        [data-testid="stDataFrame"] td { font-size: 15px !important; }
        [data-testid="stMetricValue"] { font-size: 13px !important; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🌅 Filter Average 4 Periods EMA Ribbon</h1>", unsafe_allow_html=True)

# --- PATH LOGIC ---
LOCAL_PATH = r"C:\Users\swong\Owner\Python\US Trading\Ticker"
TICKER_FOLDER = LOCAL_PATH if os.path.exists(LOCAL_PATH) else "ticker"

COLUMN_ORDER = [
    'Ticker', 'Price Now', 'Last', 'Days', 'Trend', 'Last-5D', 'TD9 Up', 'TD9 Down',
    'Distance%', 'P-Gap%', '5D-MA', '20D-MA', '50D-MA', 
    'Cross', 'AVG EMA'
]

# --- Technical Calculations ---
def calculate_kama(series, period=10, fast=2, slow=30):
    series = series.squeeze()
    change = abs(series.diff(period))
    volatility = abs(series.diff()).rolling(window=period).sum()
    er = (change / volatility).fillna(0)
    sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
    kama = np.zeros_like(series)
    start_idx = min(period, len(series)-1)
    if len(series) > start_idx:
        kama[start_idx] = series.iloc[start_idx]
        for i in range(start_idx + 1, len(series)):
            kama[i] = kama[i-1] + sc.iloc[i] * (series.iloc[i] - kama[i-1])
    return pd.Series(kama, index=series.index)

def get_crossover_trend_stats(df, live_price):
    ema = df['AVG_EMA'].squeeze()
    ma20 = df['20MA'].squeeze()
    close = df['Close'].squeeze()
    is_above = ema > ma20
    current_side = bool(is_above.iloc[-1])
    diff = is_above.ne(current_side).iloc[::-1]
    days_since_cross = len(is_above.loc[diff.idxmax():]) - 1 if diff.any() else len(is_above)
    
    p_now = float(live_price)
    p_5d_ago = float(close.iloc[-5]) if len(close) > 5 else float(close.iloc[0])
    cross_start_idx = -max(1, days_since_cross)
    p_cross_day_one = float(close.iloc[cross_start_idx])
    
    perf_5d = ((p_now - p_5d_ago) / p_5d_ago) * 100
    recent_5d = "Up" if perf_5d > 0.5 else ("Down" if perf_5d < -0.5 else "Sideway")
    
    total_ret = ((p_now - p_cross_day_one) / p_cross_day_one) * 100
    gap = ((p_now - float(ema.iloc[-1])) / float(ema.iloc[-1])) * 100
    return days_since_cross, recent_5d, f"{gap:.2f}%", f"{total_ret:.2f}%"

def calculate_td9(close_series):
    close = close_series.values
    n = len(close)

    buy_cond  = [close[i] < close[i - 4] if i >= 4 else False for i in range(n)]
    sell_cond = [close[i] > close[i - 4] if i >= 4 else False for i in range(n)]

    def bars_last_count(cond):
        count = [0] * n
        for i in range(n):
            if cond[i]:
                count[i] = count[i - 1] + 1 if i > 0 else 1
            else:
                count[i] = 0
        return count

    buy_count = bars_last_count(buy_cond)
    sell_count = bars_last_count(sell_cond)

    # ✅ ONLY return latest value (like your filter needs)
    latest_buy = buy_count[-1] if 1 <= buy_count[-1] <= 9 else 0
    latest_sell = sell_count[-1] if 1 <= sell_count[-1] <= 9 else 0

    return latest_sell, latest_buy

@st.cache_data(ttl=300)
def screen_tickers(tickers):
    if not tickers: return pd.DataFrame()
    h_data = yf.download(tickers, period='3y', interval='1d', auto_adjust=False, progress=False, group_by='ticker')
    l_data = yf.download(tickers, period='1d', interval='5m', prepost=True, auto_adjust=False, progress=False, group_by='ticker')
    
    all_results = []
    for ticker in tickers:
        try:
            df = h_data[ticker].dropna(subset=['Close']) if len(tickers) > 1 else h_data
            df_live = l_data[ticker].dropna(subset=['Close']) if len(tickers) > 1 else l_data
            
            if df.empty or len(df) < 60: continue
            
            curr_p = float(df_live['Close'].iloc[-1]) if not df_live.empty else float(df['Close'].iloc[-1])
            last_p = float(df['Close'].iloc[-1])
            prev_p = float(df['Close'].iloc[-2])
            
            close_s = df['Close'].squeeze()
            td_up, td_down = calculate_td9(close_s)
            ema_all = (close_s.ewm(span=4).mean() + close_s.ewm(span=9).mean() + 
                       close_s.ewm(span=13).mean() + close_s.ewm(span=17).mean()) / 4
            df['AVG_EMA'] = ema_all.ewm(span=2).mean()
            df['5MA'] = close_s.rolling(5).mean()
            df['20MA'] = close_s.rolling(20).mean()
            df['50MA'] = close_s.rolling(50).mean()
            df['KAMA'] = calculate_kama(close_s)

            days, recent_5, gap_val, trend_ret = get_crossover_trend_stats(df, curr_p)
            l_row, p_row = df.iloc[-1], df.iloc[-2]
            
            all_results.append({
                'Ticker': ticker, 'Price Now': f"{curr_p:.2f}", 'Last': f"{last_p:.2f}",
                'Days': days, 'Trend': "Up" if curr_p > float(l_row['AVG_EMA']) else "Down", 
                'Last-5D': recent_5, 'Distance%': gap_val, 'P-Gap%': trend_ret, 
                'AVG EMA': f"{float(l_row['AVG_EMA']):.2f}",
                '5MA': f"{float(l_row['5MA']):.2f}", '20MA': f"{float(l_row['20MA']):.2f}",
                '50MA': f"{float(l_row['50MA']):.2f}", 'KAMA': f"{float(l_row['KAMA']):.2f}",
                '5D-MA': 'Above' if curr_p > float(l_row['5MA']) else 'Below',
                '20D-MA': 'Above' if curr_p > float(l_row['20MA']) else 'Below',
                '50D-MA': 'Above' if curr_p > float(l_row['50MA']) else 'Below',
                'Cross': 'Yes' if (float(p_row['AVG_EMA']) <= float(p_row['20MA']) and float(l_row['AVG_EMA']) > float(l_row['20MA'])) else '',
                'Alignment': 'AVG_EMA_Aligned' if float(l_row['AVG_EMA']) > float(l_row['20MA']) else '20_MA_Aligned',
                'TD9 Up': td_up if td_up > 0 else None,
                'TD9 Down': td_down if td_down > 0 else None
            })
        except: continue
    return pd.DataFrame(all_results)

def color_price_now(val, last):
    try:
        return "color: green;" if float(val) > float(last) else "color: red;"
    except:
        return ""

def color_ma(val, price):
    try:
        return "color: green;" if float(price) > float(val) else "color: red;"
    except:
        return ""

def color_avg_ema(val, price):
    try:
        return "color: green;" if float(val) > float(price) else "color: red;"
    except:
        return ""

# --- ORIGINAL UI LAYOUT ---
NY_TZ = ZoneInfo("America/New_York")
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"<p style='margin-bottom: 0px; color: #888; font-size: 12px;'>Enter CSV File Name | Last refresh (AMS /NY): {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    file_name = st.text_input("CSV:", value="WATCH", autocomplete="off", key="csv_input", label_visibility="collapsed").strip()
    
with col2:
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    if st.button("🔄"):
        st.cache_data.clear()
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.rerun()

csv_path = os.path.join(TICKER_FOLDER, f"{file_name}.csv")

if os.path.exists(csv_path):
    tickers_raw = pd.read_csv(csv_path).iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
    df_main = screen_tickers(tickers_raw)

    # ✅ FIX TD9 dtype (ADD HERE)
    if not df_main.empty:
        df_main['TD9 Up'] = df_main['TD9 Up'].astype('Int64')
        df_main['TD9 Down'] = df_main['TD9 Down'].astype('Int64')

    if not df_main.empty:
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Tickers", len(tickers_raw))
        m2.metric("Upward", len(df_main[df_main['Alignment'] == 'AVG_EMA_Aligned']))
        m3.metric("Downward", len(df_main[df_main['Alignment'] == '20_MA_Aligned']))

        tabs = st.tabs(["📈 Upward", "📉 Downward", "📊 All Data"])

        # =========================
        # 📈 UPWARD
        # =========================
        with tabs[0]:
            df_up = df_main[df_main['Alignment'] == 'AVG_EMA_Aligned'].sort_values('Days')
            df_up = df_up[COLUMN_ORDER]

            styled_up = df_up.style \
                .apply(lambda row: [
                    "color: green;" if col == 'Price Now' and float(row['Price Now']) > float(row['Last'])
                    else "color: red;" if col == 'Price Now'
                    else "" for col in df_up.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '5D-MA' and row['5D-MA'] == 'Above'
                    else "color: red;" if col == '5D-MA'
                    else "" for col in df_up.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '20D-MA' and row['20D-MA'] == 'Above'
                    else "color: red;" if col == '20D-MA'
                    else "" for col in df_up.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '50D-MA' and row['50D-MA'] == 'Above'
                    else "color: red;" if col == '50D-MA'
                    else "" for col in df_up.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == 'AVG EMA' and float(row['AVG EMA']) > float(row['Price Now'])
                    else "color: red;" if col == 'AVG EMA'
                    else "" for col in df_up.columns
                ], axis=1)

            st.dataframe(styled_up, width="stretch", hide_index=True)

        # =========================
        # 📉 DOWNWARD
        # =========================
        with tabs[1]:
            df_down = df_main[df_main['Alignment'] == '20_MA_Aligned'].sort_values('Days')
            df_down = df_down[COLUMN_ORDER]

            styled_down = df_down.style \
                .apply(lambda row: [
                    "color: green;" if col == 'Price Now' and float(row['Price Now']) > float(row['Last'])
                    else "color: red;" if col == 'Price Now'
                    else "" for col in df_down.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '5D-MA' and row['5D-MA'] == 'Above'
                    else "color: red;" if col == '5D-MA'
                    else "" for col in df_down.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '20D-MA' and row['20D-MA'] == 'Above'
                    else "color: red;" if col == '20D-MA'
                    else "" for col in df_down.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '50D-MA' and row['50D-MA'] == 'Above'
                    else "color: red;" if col == '50D-MA'
                    else "" for col in df_down.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == 'AVG EMA' and float(row['AVG EMA']) > float(row['Price Now'])
                    else "color: red;" if col == 'AVG EMA'
                    else "" for col in df_down.columns
                ], axis=1)

            st.dataframe(styled_down, width="stretch", hide_index=True)

        # =========================
        # 📊 ALL DATA
        # =========================
        with tabs[2]:
            df_all = df_main[COLUMN_ORDER]

            styled_all = df_all.style \
                .apply(lambda row: [
                    "color: green;" if col == 'Price Now' and float(row['Price Now']) > float(row['Last'])
                    else "color: red;" if col == 'Price Now'
                    else "" for col in df_all.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '5D-MA' and row['5D-MA'] == 'Above'
                    else "color: red;" if col == '5D-MA'
                    else "" for col in df_all.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '20D-MA' and row['20D-MA'] == 'Above'
                    else "color: red;" if col == '20D-MA'
                    else "" for col in df_all.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == '50D-MA' and row['50D-MA'] == 'Above'
                    else "color: red;" if col == '50D-MA'
                    else "" for col in df_all.columns
                ], axis=1) \
                .apply(lambda row: [
                    "color: green;" if col == 'AVG EMA' and float(row['AVG EMA']) > float(row['Price Now'])
                    else "color: red;" if col == 'AVG EMA'
                    else "" for col in df_all.columns
                ], axis=1)

            st.dataframe(styled_all, width="stretch", hide_index=True)

else:
    st.error(f"File not found: {csv_path}")
