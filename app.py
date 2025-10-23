# -*- coding: utf-8 -*-
# Baccarat Ultimate Pro - 完整手机优化版
import streamlit as st
import numpy as np
import pandas as pd
import math
import re
from collections import defaultdict

st.set_page_config(page_title="百家乐终极版", layout="centered")

# 手机优化CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        margin: 5px 0;
    }
    .big-text { font-size: 24px !important; }
    .card-input { 
        background: #2d3748;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 style="text-align:center;color:#FFD700;">🎯 百家乐终极版</h1>', unsafe_allow_html=True)

# 状态管理
if "games" not in st.session_state:
    st.session_state.games = []
if "roads" not in st.session_state:
    st.session_state.roads = {'bead_road': [], 'big_road': []}

# 快速输入系统
st.markdown("## ⌨️ 快速输入")
col1, col2 = st.columns(2)
with col1:
    player_cards = st.text_input("闲家牌", placeholder="例: K10 或 552", key="player")
with col2:
    banker_cards = st.text_input("庄家牌", placeholder="例: 55 或 AJ", key="banker")

# 大按钮结果选择
st.markdown("## 🏆 选择结果")
col1, col2, col3 = st.columns(3)
with col1:
    banker_btn = st.button("🔴 庄赢", use_container_width=True)
with col2:
    player_btn = st.button("🔵 闲赢", use_container_width=True)
with col3:
    tie_btn = st.button("⚪ 和局", use_container_width=True)

# 解析牌点函数
def parse_cards(input_str):
    if not input_str: return []
    input_str = input_str.upper().replace(' ', '')
    cards = []
    i = 0
    while i < len(input_str):
        if i+1 < len(input_str) and input_str[i:i+2] == '10':
            cards.append('10'); i += 2
        elif input_str[i] in ['1','2','3','4','5','6','7','8','9']:
            cards.append(input_str[i]); i += 1
        elif input_str[i] in ['A','J','Q','K','0']:
            card_map = {'A':'A', 'J':'J', 'Q':'Q', 'K':'K', '0':'10'}
            cards.append(card_map[input_str[i]]); i += 1
        else: i += 1
    return cards

# 记录游戏
def record_game():
    p_cards = parse_cards(player_cards)
    b_cards = parse_cards(banker_cards)
    
    if len(p_cards) < 2 or len(b_cards) < 2:
        st.error("需要至少2张牌")
        return
        
    result = None
    if banker_btn: result = 'B'
    elif player_btn: result = 'P'  
    elif tie_btn: result = 'T'
    
    if result:
        game_data = {
            'round': len(st.session_state.games) + 1,
            'player_cards': p_cards,
            'banker_cards': b_cards, 
            'result': result
        }
        st.session_state.games.append(game_data)
        
        # 更新路子
        if result in ['B','P']:
            st.session_state.roads['bead_road'].append(result)
            
        st.success(f"记录成功! 闲{'-'.join(p_cards)} 庄{'-'.join(b_cards)} → {result}")
        st.rerun()

if banker_btn or player_btn or tie_btn:
    record_game()

# 分析系统
if st.session_state.games:
    st.markdown("## 📊 分析结果")
    bead_road = st.session_state.roads['bead_road']
    
    if bead_road:
        # 显示珠路
        st.write("珠路:", " ".join(["🔴" if x=='B' else "🔵" for x in bead_road[-10:]]))
        
        # 简单分析
        current = bead_road[-1] if bead_road else None
        streak = 1
        for i in range(len(bead_road)-2, -1, -1):
            if bead_road[i] == current: streak += 1
            else: break
            
        if streak >= 3:
            st.info(f"当前趋势: {current} (连{streak}局)")
            st.metric("推荐方向", current, f"连赢{streak}局")
        else:
            st.info("趋势不明显，建议观察")

# 历史记录
if st.session_state.games:
    st.markdown("## 📝 最近牌局")
    for game in list(reversed(st.session_state.games))[-5:]:
        st.write(f"第{game['round']}局: 闲{'-'.join(game['player_cards'])} 庄{'-'.join(game['banker_cards'])} → "
                f"{'庄赢' if game['result']=='B' else '闲赢' if game['result']=='P' else '和局'}")

# 重置按钮
if st.button("🔄 开始新牌靴", use_container_width=True):
    st.session_state.games.clear()
    st.session_state.roads = {'bead_road': [], 'big_road': []}
    st.success("新牌靴开始!")
    st.rerun()
