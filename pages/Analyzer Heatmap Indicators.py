import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from zoneinfo import ZoneInfo

# =====================================================
# CONFIG
# =====================================================
NY_TZ = ZoneInfo("America/New_York")
TIMEFRAMES = [
    ("Ultra-Short", 5, 10),
    ("Short", 10, 20),
    ("Swing", 20, 50),
    ("Position", 50, 200),
    ("Day", 1, 5),
]

st.set_page_config(layout="wide", page_title="Indicators Heatmap")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.25rem;
            padding-left: 1.75rem;
            padding-right: 1.75rem;
            padding-bottom: 0rem;
        }
        html, body, [class*="css"] { font-size: 13px !important; }
        h1 { font-size: 21px !important; margin-bottom: 8px !important; }
        .flag-box {
            min-height: 86px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.14);
            color: #111;
            text-align: center;
            font-weight: 700;
        }
        .flag-label { font-size: 12px; opacity: 0.78; margin-bottom: 4px; }
        .flag-title { font-size: 15px; line-height: 1.15; }
        .flag-detail { font-size: 11px; line-height: 1.2; margin-top: 5px; font-weight: 600; }
        .small-note { color: #999; font-size: 12px; margin: 0 0 8px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>Indicators Heatmap</h1>", unsafe_allow_html=True)

plt.rcParams.update({
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "legend.fontsize": 6,
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#444",
    "xtick.color": "#d0d0d0",
    "ytick.color": "#d0d0d0",
    "grid.color": "#333",
    "text.color": "white",
})

# =====================================================
# DATA
# =====================================================
@st.cache_data(ttl=900)
def load_price_data(symbol):
    df = yf.download(symbol, period="1y", interval="1d", progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]


# =====================================================
# INDICATORS
# =====================================================
def calc_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean().replace(0, 1e-9)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_mfi(df, period):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    money_flow = typical_price * df["Volume"]
    direction = typical_price.diff()
    positive_flow = money_flow.where(direction > 0, 0)
    negative_flow = money_flow.where(direction < 0, 0)
    positive_sum = positive_flow.ewm(alpha=1 / period, adjust=False).mean()
    negative_sum = negative_flow.ewm(alpha=1 / period, adjust=False).mean().replace(0, 1e-9)
    ratio = positive_sum / negative_sum
    return 100 - (100 / (1 + ratio))


def calc_cci(df, period):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    avg = typical_price.ewm(alpha=1 / period, adjust=False).mean()
    mean_dev = (typical_price - avg).abs().ewm(alpha=1 / period, adjust=False).mean()
    return (typical_price - avg) / (0.015 * mean_dev.replace(0, 1e-9))


def calc_obv(df):
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def calc_kama(close, period=10, fast=2, slow=30):
    price = pd.Series(close).astype(float)
    change = (price - price.shift(period)).abs()
    volatility = price.diff().abs().rolling(period).sum().replace(0, np.nan)
    er = change / volatility

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = np.zeros(len(price))
    initial = price.iloc[: max(period, 1)].mean()
    kama[:period] = initial
    price_values = price.values
    sc_values = sc.values

    for i in range(period, len(price)):
        sc_value = 0 if np.isnan(sc_values[i]) else sc_values[i]
        kama[i] = kama[i - 1] + sc_value * (price_values[i] - kama[i - 1])

    return pd.Series(kama, index=price.index)


def calc_kdj(df, period):
    low_min = df["Low"].rolling(period, min_periods=1).min()
    high_max = df["High"].rolling(period, min_periods=1).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min + 1e-9) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_macd(df, fast, slow, signal):
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal


def calc_wvad_raw(df):
    day_range = (df["High"] - df["Low"]).replace(0, 1e-9)
    return df["Volume"] * (df["Close"] - df["Open"]) / day_range


def add_indicators(raw_df, short_period, long_period):
    df = raw_df.copy()
    close = df["Close"]

    df["EMA_S"] = close.ewm(span=short_period, adjust=False).mean()
    df["EMA_L"] = close.ewm(span=long_period, adjust=False).mean()
    df["KAMA_S"] = calc_kama(close, short_period)
    df["KAMA_L"] = calc_kama(close, long_period)
    df["KAMA_S_ANGLE"] = np.degrees(np.arctan(df["KAMA_S"].diff(short_period) / max(short_period, 1)))
    df["KAMA_L_ANGLE"] = np.degrees(np.arctan(df["KAMA_L"].diff(long_period) / max(long_period, 1)))

    df["RSI_S"] = calc_rsi(close, short_period)
    df["RSI_L"] = calc_rsi(close, long_period)
    df["MFI_S"] = calc_mfi(df, short_period)
    df["MFI_L"] = calc_mfi(df, long_period)
    df["CCI_S"] = calc_cci(df, short_period)
    df["CCI_L"] = calc_cci(df, long_period)
    df["MOM_S"] = close.pct_change(short_period) * 100
    df["MOM_L"] = close.pct_change(long_period) * 100

    log_ret = np.log(close / close.shift(1))
    df["HV_S"] = log_ret.ewm(span=short_period, adjust=False).std() * np.sqrt(252) * 100
    df["HV_L"] = log_ret.ewm(span=long_period, adjust=False).std() * np.sqrt(252) * 100

    obv = calc_obv(df)
    df["OBV_S"] = obv.rolling(short_period).mean() / 1e6
    df["OBV_L"] = obv.rolling(long_period).mean() / 1e6

    wvad = calc_wvad_raw(df)
    df["WVAD_S"] = wvad.rolling(short_period).sum() / 1e6
    df["WVAD_L"] = wvad.rolling(long_period).sum() / 1e6
    df["WVAD_Z"] = (df["WVAD_S"] - df["WVAD_S"].rolling(60).mean()) / df["WVAD_S"].rolling(60).std().replace(0, 1e-9)

    df["TOR_S"] = df["Volume"] / df["Volume"].rolling(short_period).mean().replace(0, 1e-9)
    df["TOR_L"] = df["Volume"] / df["Volume"].rolling(long_period).mean().replace(0, 1e-9)

    df["K"], df["D"], df["J"] = calc_kdj(df, short_period)
    df["MACD"], df["MACD_SIGNAL"], df["MACD_HIST"] = calc_macd(df, short_period, long_period, max(int(long_period / 2), 2))

    df["BIAS_S"] = (close - df["EMA_S"]) / df["EMA_S"] * 100
    df["BIAS_L"] = (close - df["EMA_L"]) / df["EMA_L"] * 100
    df["ATR_PCT"] = (df["High"] - df["Low"]).rolling(14).mean() / close * 100
    return df

# =====================================================
# SENTIMENT ENGINE
# =====================================================
def score_diff(short_value, long_value, soft=1, hard=5):
    diff = short_value - long_value
    if diff >= hard:
        return 2
    if diff >= soft:
        return 1
    if diff <= -hard:
        return -2
    if diff <= -soft:
        return -1
    return 0

def sentiment_for(df, short_period, long_period):
    ind = add_indicators(df, short_period, long_period).dropna(subset=["Close", "KAMA_L", "RSI_S", "RSI_L", "MFI_S", "MFI_L"])
    if len(ind) < 2:
        return {
            "title": "NO EDGE",
            "detail": "Not enough clean data",
            "color": "#888888",
            "score": 0,
        }

    latest = ind.iloc[-1]
    prev = ind.iloc[-2]

    momentum_score = score_diff(latest["RSI_S"], latest["RSI_L"]) + score_diff(latest["MFI_S"], latest["MFI_L"])
    trend_pct = (latest["KAMA_L"] - prev["KAMA_L"]) / max(abs(prev["KAMA_L"]), 1e-9) * 100
    bias = latest["BIAS_L"]
    wvad = latest["WVAD_S"]
    wvad_z = latest["WVAD_Z"] if not np.isnan(latest["WVAD_Z"]) else 0
    atr_pct = latest["ATR_PCT"] if not np.isnan(latest["ATR_PCT"]) else 0

    uptrend = trend_pct > 0.05
    downtrend = trend_pct < -0.05
    flat = not uptrend and not downtrend
    supported_volume = wvad > 0 and wvad_z > -0.5
    distribution = wvad < 0 and wvad_z < 0.5
    extended = bias > max(3, atr_pct * 1.25)
    oversold_bounce_area = bias < -max(2, atr_pct)

    if uptrend and momentum_score >= 2 and supported_volume and not extended:
        return {
            "title": "TREND SUPPORTED",
            "detail": "KAMA Rise | BIAS Controlled | WVAD Positive | SCORE Expanding",
            "color": "#00d084",
            "score": momentum_score,
        }

    if uptrend and extended and supported_volume:
        return {
            "title": "EXTENDED UPTREND",
            "detail": "KAMA Rise | BIAS Limit | WVAD Positive | SCORE Elevated",
            "color": "#ffd54f",
            "score": momentum_score,
        }

    if uptrend and distribution:
        return {
            "title": "UPTREND LOSING VOLUME",
            "detail": "KAMA Rise | BIAS Elevated | WVAD Negative | SCORE Fading",
            "color": "#ffb74d",
            "score": momentum_score,
        }

    if flat and supported_volume and momentum_score >= 0:
        return {
            "title": "ACCUMULATION WATCH",
            "detail": "KAMA Coiling | BIAS Stable | WVAD Positive | SCORE Improve",
            "color": "#64d8ff",
            "score": momentum_score,
        }

    if downtrend and momentum_score <= -2 and distribution:
        return {
            "title": "DOWNTREND CONFIRMED",
            "detail": "KAMA Fall | BIAS Weak | WVAD Negative | SCORE Drop",
            "color": "#ff5252",
            "score": momentum_score,
        }

    if downtrend and supported_volume and momentum_score > 0:
        return {
            "title": "RECOVERY WATCH",
            "detail": "KAMA Fall | BIAS Recover | WVAD Positive | SCORE Improve",
            "color": "#ef9a9a",
            "score": momentum_score,
        }

    if oversold_bounce_area and momentum_score > -2:
        return {
            "title": "PULLBACK WATCH",
            "detail": "KAMA Weak | BIAS Compress | WVAD Stable | SCORE Recover",
            "color": "#b2dfdb",
            "score": momentum_score,
        }

    return {
        "title": "NO DIRECTION",
        "detail": "NO DIRECTION",
        "color": "#b0bec5",
        "score": momentum_score,
    }


# =====================================================
# DISPLAY HELPERS
# =====================================================
def flag_box(label, flag):
    return f"""
    <div class="flag-box" style="background:{flag['color']}">
        <div class="flag-label">{label}</div>
        <div class="flag-title">{flag['title']}</div>
        <div class="flag-detail">{flag['detail']}</div>
    </div>
    """

def color_scale(value, red, pink, lime, green):
    try:
        val = float(value)
    except (TypeError, ValueError):
        return ""
    if val <= red:
        return "background-color:#b71c1c;color:white;"
    if val <= pink:
        return "background-color:#ffcdd2;color:#111;"
    if val >= green:
        return "background-color:#1b5e20;color:white;"
    if val >= lime:
        return "background-color:#c8e6c9;color:#111;"
    return ""

def style_table(table, short_period, long_period):
    styled = table.style
    styled = styled.map(lambda v: color_scale(v, 20, 40, 60, 80), subset=["RSI_S", "RSI_L", "MFI_S", "MFI_L"])
    styled = styled.map(lambda v: color_scale(v, -120, -80, 80, 120), subset=["CCI_S", "CCI_L"])
    styled = styled.map(lambda v: color_scale(v, -5, -2, 2, 5), subset=["MOM_S", "MOM_L"])
    styled = styled.map(lambda v: color_scale(v, -3, -1, 1, 3), subset=["BIAS_S", "BIAS_L"])
    styled = styled.map(lambda v: color_scale(v, -20, -8, 8, 20), subset=["KAMA_S_ANGLE", "KAMA_L_ANGLE"])
    styled = styled.map(lambda v: color_scale(v, -2, -0.5, 0.5, 2), subset=["WVAD_S", "WVAD_L", "WVAD_Z"])
    styled = styled.map(lambda v: color_scale(v, 0.6, 0.85, 1.15, 1.6), subset=["TOR_S", "TOR_L"])
    styled = styled.map(lambda v: color_scale(v, 8, 18, 35, 55), subset=["HV_S", "HV_L"])
    styled = styled.map(lambda v: color_scale(v, -1, -0.1, 0.1, 1), subset=["MACD_HIST"])
    styled = styled.map(lambda v: color_scale(v, -2, -0.5, 0.5, 2), subset=["MACD", "MACD_SIGNAL"])
    styled = styled.map(lambda v: color_scale(v, 20, 40, 60, 80), subset=["K", "D"])
    styled = styled.map(lambda v: color_scale(v, 10, 30, 70, 90), subset=["J"])
    def obv_logic(v):
        try:
            val = float(v)
            if val > 0: return "background-color:#1b5e20;color:white;"
            if val < 0: return "background-color:#b71c1c;color:white;"
        except: pass
        return ""    
    styled = styled.map(obv_logic, subset=["OBV_S", "OBV_L"])
    return styled

def render_mini_charts(df, short_period, long_period):

    pdf = df.tail(90).copy()
    latest = df.iloc[-1]

    def add_panel_label(fig_obj, row, text):
        axis_name = "yaxis" if row == 1 else f"yaxis{row}"
        domain = getattr(fig_obj.layout, axis_name).domain

        fig_obj.add_annotation(
            text=text,
            x=0.5,
            y=domain[1] - 0.01,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=13, color="white"),
            bgcolor="rgba(14,17,23,0.65)",
            borderpad=2,
        )

    left_col, right_col = st.columns(2)

    # =====================================================
    # LEFT SIDE
    # =====================================================
    with left_col:
        fig_left = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035)
        # HV
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["HV_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=1, col=1)
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["HV_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=1, col=1)
        add_panel_label(fig_left, 1, f"HV{short_period}: {latest['HV_S']:.2f}% | HV{long_period}: {latest['HV_L']:.2f}%")

        # CCI
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["CCI_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=2, col=1)
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["CCI_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=2, col=1)
        fig_left.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=2, col=1)
        add_panel_label(fig_left, 2, f"CCI{short_period}: {latest['CCI_S']:.2f} | CCI{long_period}: {latest['CCI_L']:.2f}%")

        # MOM
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["MOM_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=3, col=1)
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["MOM_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=3, col=1)
        fig_left.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=3, col=1)
        add_panel_label(fig_left, 3, f"MOM{short_period}: {latest['MOM_S']:.2f}% | MOM{long_period}: {latest['MOM_L']:.2f}%")

        # ANGLE
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["KAMA_S_ANGLE"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=4, col=1)
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["KAMA_L_ANGLE"], line=dict(color="lightgreen", width=1), showlegend=False), row=4, col=1)
        fig_left.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=4, col=1)
        add_panel_label(fig_left, 4, f"ANG{short_period}: {latest['KAMA_S_ANGLE']:.2f} | ANG{long_period}: {latest['KAMA_L_ANGLE']:.2f}")

        # OBV
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["OBV_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=5, col=1)
        fig_left.add_trace(go.Scatter(x=pdf.index, y=pdf["OBV_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=5, col=1)
        add_panel_label(fig_left, 5, f"OBV{short_period}: {latest['OBV_S']:.2f}M | OBV{long_period}: {latest['OBV_L']:.2f}M")

        fig_left.update_layout(
            height=1000,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=20, r=15, t=40, b=20),
            font=dict(size=13),
        )

        st.plotly_chart(fig_left, width="stretch")

    # =====================================================
    # RIGHT SIDE
    # =====================================================
    with right_col:
        fig_right = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.035)

        # TOR
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["TOR_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=1, col=1)
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["TOR_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=1, col=1)
        fig_right.add_hline(y=1, line_dash="dash", line_color="gray", line_width=0.5, row=1, col=1)
        add_panel_label(fig_right, 1, f"TOR{short_period}: {latest['TOR_S']:.2f}x | TOR{long_period}: {latest['TOR_L']:.2f}x")

        # MFI
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["MFI_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=2, col=1)
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["MFI_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=2, col=1)
        fig_right.add_hline(y=80, line_dash="dash", line_color="green", line_width=0.5, row=2, col=1)
        fig_right.add_hline(y=20, line_dash="dash", line_color="red", line_width=0.5, row=2, col=1)
        add_panel_label(fig_right, 2, f"MFI{short_period}: {latest['MFI_S']:.2f} | MFI{long_period}: {latest['MFI_L']:.2f}")

        # RSI
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["RSI_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=3, col=1)
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["RSI_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=3, col=1)
        fig_right.add_hline(y=80, line_dash="dash", line_color="green", line_width=0.5, row=3, col=1)
        fig_right.add_hline(y=20, line_dash="dash", line_color="red", line_width=0.5, row=3, col=1)
        add_panel_label(fig_right, 3, f"RSI{short_period}: {latest['RSI_S']:.2f} | RSI{long_period}: {latest['RSI_L']:.2f}")

        # BIAS
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["BIAS_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=4, col=1)
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["BIAS_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=4, col=1)
        fig_right.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=4, col=1)
        add_panel_label(fig_right, 4, f"BIAS{short_period}: {latest['BIAS_S']:.2f}% | BIAS{long_period}: {latest['BIAS_L']:.2f}%")

        # WVAD
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["WVAD_S"], line=dict(color="white", dash="dot", width=1), showlegend=False), row=5, col=1)
        fig_right.add_trace(go.Scatter(x=pdf.index, y=pdf["WVAD_L"], line=dict(color="lightgreen", width=1), showlegend=False), row=5, col=1)
        fig_right.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5, row=5, col=1)
        add_panel_label(fig_right, 5, f"WVAD{short_period}: {latest['WVAD_S']:.2f}M | WVAD{long_period}: {latest['WVAD_L']:.2f}M")

        fig_right.update_layout(
            height=1000,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=20, r=15, t=40, b=20),
            font=dict(size=13),
        )

        st.plotly_chart(fig_right, width="stretch")

def render_timeframe(raw_df, label, short_period, long_period):
    df = add_indicators(raw_df, short_period, long_period).dropna(subset=["Close"])
    if len(df) < 2:
        st.warning("Not enough data for this timeframe.")
        return

    latest = df.iloc[-1]
    table_cols = [
        "HV_S", "HV_L",
        "CCI_S", "CCI_L",
        "MFI_S", "MFI_L",
        "MOM_S", "MOM_L",
        "RSI_S", "RSI_L",
        "KAMA_S_ANGLE", "KAMA_L_ANGLE",
        "BIAS_S", "BIAS_L",
        "OBV_S", "OBV_L",
        "MACD", "MACD_SIGNAL", "MACD_HIST",
        "K", "D", "J",
        "TOR_S", "TOR_L",
        "WVAD_S", "WVAD_L", "WVAD_Z",
    ]

    view = df[table_cols].tail(20).sort_index(ascending=False).round(2)
    view.index = view.index.strftime("%Y-%m-%d")

    st.dataframe(
        style_table(view, short_period, long_period).format({
            "HV_S": "{:.0f}%", "HV_L": "{:.0f}%",
            "CCI_S": "{:.0f}", "CCI_L": "{:.0f}",
            "MFI_S": "{:.0f}", "MFI_L": "{:.0f}",
            "MOM_S": "{:.1f}%", "MOM_L": "{:.1f}%",
            "RSI_S": "{:.0f}", "RSI_L": "{:.0f}",
            "KAMA_S_ANGLE": "{:.1f}", "KAMA_L_ANGLE": "{:.1f}",
            "BIAS_S": "{:.1f}%", "BIAS_L": "{:.1f}%",
            "OBV_S": "{:.1f}M", "OBV_L": "{:.1f}M",
            "MACD": "{:.2f}", "MACD_SIGNAL": "{:.2f}", "MACD_HIST": "{:.2f}",
            "K": "{:.0f}", "D": "{:.0f}", "J": "{:.0f}",
            "TOR_S": "{:.2f}x", "TOR_L": "{:.2f}x",
            "WVAD_S": "{:.2f}M", "WVAD_L": "{:.2f}M", "WVAD_Z": "{:.2f}",
        }),
        height=340,
        width="stretch",
        column_config={
            "MACD": "DIF",
            "MACD_SIGNAL": "DEA",
            "MACD_HIST": "MACD",
            "KAMA_S_ANGLE": "ANGLE_S",
            "KAMA_L_ANGLE": "ANGLE_L"
        }
    )

    chart_df = df.tail(90)
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.4, 0.3, 0.3],
    )

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["Close"], name=f"Close {latest['Close']:.2f}", line=dict(color="#42a5f5", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA_S"], name=f"EMA{short_period}", line=dict(color="#ffffff", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["EMA_L"], name=f"EMA{long_period}", line=dict(color="#9ccc65", width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["K"], name="K", line=dict(width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["D"], name="D", line=dict(width=1, color="#ffb74d")), row=2, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["J"], name="J", line=dict(width=1, color="#ffffff")), row=2, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="#66bb6a", row=2, col=1)

    macd_colors = np.where(chart_df["MACD_HIST"] >= 0, "#43a047", "#e53935")
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["MACD_HIST"], marker_color=macd_colors, name="MACD Hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"], name="DIF", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_SIGNAL"], name="DEA", line=dict(width=1, color="#ffffff")), row=3, col=1)


    fig.update_layout(
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=35, b=20),
        #title=f"{label} ({short_period}D / {long_period}D)",
    )
    st.plotly_chart(fig, width="stretch")
    render_mini_charts(df, short_period, long_period)

# =====================================================
# APP
# =====================================================
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now(NY_TZ)

top_left, top_right = st.columns([1, 0.5])
with top_left:
    st.markdown(
    f"<p style='margin-bottom: 3px; color: #888; font-size: 12px;'>"
    f"Enter Stock Ticker  |  Last refresh (AMS / NY) : "
    f"{st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
    )
    ticker = st.text_input("Ticker", value="APP", autocomplete="off", label_visibility="collapsed").strip().upper()

with top_right:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if st.button("🔄",width="stretch"):
        st.session_state.last_refresh = datetime.now(NY_TZ)
        st.cache_data.clear()
        st.rerun()

if not ticker:
    st.info("Enter a ticker to start.")
    st.stop()

raw = load_price_data(ticker)
if raw.empty:
    st.error(f"No price data returned for '{ticker}'.")
    st.stop()

if len(raw) < 60:
    st.warning("This ticker has limited price history. Longer-term flags may be unreliable.")

flag_cols = st.columns(4)
for col, (label, short_period, long_period) in zip(flag_cols, TIMEFRAMES[:4]):
    with col:
        st.markdown(flag_box(f"{label} {short_period}/{long_period}", sentiment_for(raw, short_period, long_period)), unsafe_allow_html=True)

tabs = st.tabs([f"{label} ({short_period}/{long_period})" for label, short_period, long_period in TIMEFRAMES] + ["Guide"])
for tab, (label, short_period, long_period) in zip(tabs[:-1], TIMEFRAMES):
    with tab:
        render_timeframe(raw, label, short_period, long_period)

with tabs[-1]:
    st.markdown(
        """
        ### How to read the sentiment flags

        **TREND SUPPORTED** means the stock is trending up, short-term RSI/MFI are leading the longer timeframe, and WVAD confirms positive volume pressure. This is a watchlist-quality bullish condition, not an automatic buy.

        **EXTENDED UPTREND** means the trend is still strong but price is stretched versus its average and recent range. Traders usually wait for a pullback, consolidation, or clean breakout trigger.

        **UPTREND LOSING VOLUME** means price remains strong but WVAD has turned negative. That is a caution flag for chasing.

        **ACCUMULATION WATCH** means price trend is flat but volume pressure is improving. This is useful for early watchlist building.

        **DOWNTREND CONFIRMED** means trend, momentum, and volume pressure are aligned bearish.

        **RECOVERY WATCH** means the stock is still in a downtrend, but momentum and volume pressure are improving. Treat it as early evidence only.

        Best use: let the sentiment flag define the market condition, then use your own setup trigger for entry, stop, and position size.
        """
    )
