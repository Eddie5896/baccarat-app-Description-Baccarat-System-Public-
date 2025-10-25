# -*- coding: utf-8 -*-
# Baccarat Master Ultimate - Precision 13.5 Ultimate · EOR Fusion 版
# 界面优化版 - 现代化科技感设计

import streamlit as st
import numpy as np
import math
import json
from collections import defaultdict, Counter
from datetime import datetime
from itertools import groupby
import plotly.graph_objects as go
import plotly.express as px

# ========================== 基础配置 ==========================
st.set_page_config(
    page_title="🐉 百家乐大师 Precision 13.5 · EOR Fusion", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 现代化CSS样式
st.markdown("""
<style>
    /* 主色调：深蓝科技风 */
    .main-header {
        font-size: 2.5rem;
        color: #00D4FF;
        text-align: center;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        margin-bottom: 1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00D4FF, #0099CC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 卡片样式 */
    .modern-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.2);
    }
    
    /* 预测卡片 */
    .prediction-card {
        background: linear-gradient(135deg, #0F172A, #1E293B);
        border: 2px solid #00D4FF;
        border-radius: 16px;
        padding: 25px;
        margin: 15px 0;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }
    
    /* 指标卡片 */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(100, 116, 139, 0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    
    /* 按钮样式 */
    .stButton>button {
        background: linear-gradient(135deg, #00D4FF, #0099CC);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.4);
    }
    
    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D4FF, #0099CC);
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00D4FF, #0099CC);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #0F172A, #1E293B);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐉 百家乐大师 Precision 13.5 · EOR Fusion</h1>', unsafe_allow_html=True)

# ========================== 状态初始化 ==========================
def _init_state():
    ss = st.session_state
    ss.setdefault("ultimate_games", [])
    ss.setdefault("expert_roads", {'big_road':[],'bead_road':[],
