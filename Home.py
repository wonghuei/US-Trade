import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz

# ============================================
# PAGE CONFIG & COMPACT CSS
# ============================================
NY_TZ = pytz.timezone("America/New_York")

st.set_page_config(layout="wide", page_title="Home")

st.markdown("""
    <style>
        .block-container { padding-top: 1.5rem; padding-bottom: 0rem; padding-left: 2rem; padding-right: 2rem; }
        h1 { font-size: 28px !important; margin-bottom: 10px !important; }
        [data-testid="stMetricValue"] { font-size: 14px !important; font-weight: 700; color: #00ff00; }
        [data-testid="stMetricLabel"] { font-size: 12px !important; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>US STOCK TRADING DASHBOARD</h1>", unsafe_allow_html=True)


