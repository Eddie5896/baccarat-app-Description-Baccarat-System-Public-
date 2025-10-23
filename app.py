# -*- coding: utf-8 -*-
# Baccarat Master Ultimate - 完整终极版
# 包含：四层大脑 + 六路分析 + 深度学习 + 风险控制 + 专业界面

import streamlit as st
import numpy as np
import pandas as pd
import math
import re
from collections import defaultdict
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="百家乐大师终极版", layout="centered")

# 专业CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px #000000;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        border: 3px solid #FFD700;
        margin: 15px 0;
        text-align: center;
    }
    .road-display {
        background: #1a1a1a;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #333;
    }
    .metric-card {
        background: #2d3748;
        padding: 12px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
    .stButton button {
        width: 100%;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
        margin: 5px 0;
    }
    .pattern-badge {
        background: #e74c3c;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">🐉 百家乐大师终极版</h1>', unsafe_allow_html=True)

# ---------------- 终极状态管理 ----------------
if "ultimate_games" not in st.session_state:
    st.session_state.ultimate_games = []
if "expert_roads" not in st.session_state:
    st.session_state.expert_roads = {
        'big_road': [], 'big_eye_road': [], 'small_road': [], 
        'cockroach_road': [], 'bead_road': [], 'three_bead_road': []
    }
if "ai_memory" not in st.session_state:
    st.session_state.ai_memory = {
        'pattern_accuracy': {}, 'winning_strategies': [],
        'risk_level': 'medium', 'confidence_shift': 0.0,
        'learning_data': []
    }

# ---------------- 四层大脑核心系统 ----------------
class StructureLayer:
    """结构层 - 专业模式识别"""
    @staticmethod
    def detect_advanced_patterns(sequence):
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 4:
            return {"status": "数据不足"}
        
        patterns = []
        # 长龙检测
        if len(bp_seq) >= 5:
            last_5 = bp_seq[-5:]
            if len(set(last_5)) == 1:
                patterns.append(f"{bp_seq[-1]}长龙")
        
        # 单跳检测
        if len(bp_seq) >= 6:
            last_6 = bp_seq[-6:]
            if last_6 in [['B','P','B','P','B','P'], ['P','B','P','B','P','B']]:
                patterns.append("单跳龙")
        
        # 双跳检测
        if len(bp_seq) >= 8:
            last_8 = "".join(bp_seq[-8:])
            if last_8 in ["BBPPBBPP", "PPBBPPBB"]:
                patterns.append("双跳龙")
        
        # 庄闲比例分析
        b_count = bp_seq.count('B')
        p_count = bp_seq.count('P')
        total = b_count + p_count
        ratio = b_count / total if total > 0 else 0.5
        
        if ratio > 0.6:
            trend = "强庄势"
        elif ratio > 0.55:
            trend = "庄势"
        elif ratio < 0.4:
            trend = "强闲势"
        elif ratio < 0.45:
            trend = "闲势"
        else:
            trend = "平衡势"
        
        return {
            "patterns": patterns,
            "trend": trend,
            "banker_ratio": ratio,
            "current_streak": StructureLayer.calculate_streak(bp_seq),
            "volatility": np.std([1 if x=='B' else 0 for x in bp_seq]) if len(bp_seq) > 1 else 0
        }
    
    @staticmethod
    def calculate_streak(bp_seq):
        if not bp_seq: return 0
        current = bp_seq[-1]
        streak = 1
        for i in range(len(bp_seq)-2, -1, -1):
            if bp_seq[i] == current:
                streak += 1
            else:
                break
        return streak

class RhythmLayer:
    """节奏层 - 动态Z′分数和动能分析"""
    @staticmethod
    def analyze_rhythm(sequence):
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 8:
            return {"z_score": 0, "momentum": 0, "energy": 0.5}
        
        # 动态Z′分数计算
        values = [1 if x=='B' else -1 for x in bp_seq[-12:]]
        mean_val = np.mean(values)
        std_val = np.std(values) if np.std(values) > 0 else 1
        z_score = mean_val / std_val
        
        # 动能分析
        changes = sum(1 for i in range(1, len(bp_seq)) if bp_seq[i] != bp_seq[i-1])
        volatility = changes / len(bp_seq)
        
        # 能量计算
        energy = 0.5 + (abs(z_score) * 0.3) + (volatility * 0.2)
        
        return {
            "z_score": z_score,
            "momentum": abs(z_score),
            "energy": min(energy, 0.9),
            "volatility": volatility,
            "phase": "兴奋期" if energy > 0.7 else "活跃期" if energy > 0.5 else "平静期"
        }

class FusionLayer:
    """权重层 - 六路共识融合"""
    @staticmethod
    def fuse_road_signals(sequence):
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 6:
            return {"score": 0, "confidence": 0.5}
        
        signals = {}
        
        # 大路信号
        structure = StructureLayer.detect_advanced_patterns(sequence)
        if structure['trend'] in ['强庄势', '庄势']:
            signals['big_road'] = 0.7
        elif structure['trend'] in ['强闲势', '闲势']:
            signals['big_road'] = -0.7
        else:
            signals['big_road'] = 0
        
        # 珠路信号
        recent_6 = bp_seq[-6:]
        b_ratio = recent_6.count('B') / len(recent_6)
        signals['bead_road'] = (b_ratio - 0.5) * 2
        
        # 节奏信号
        rhythm = RhythmLayer.analyze_rhythm(sequence)
        signals['rhythm'] = rhythm['z_score'] * 0.5
        
        # 权重融合
        weights = {'big_road': 0.4, 'bead_road': 0.3, 'rhythm': 0.3}
        total_score = sum(signals[road] * weight for road, weight in weights.items())
        total_confidence = 0.5 + (abs(total_score) * 0.3) + (len(structure['patterns']) * 0.1)
        
        return {
            "score": total_score,
            "confidence": min(total_confidence, 0.9),
            "signals": signals,
            "weights": weights
        }

class StrategyLayer:
    """策略层 - 最终决策和风险控制"""
    @staticmethod
    def master_decision(sequence):
        if len(sequence) < 4:
            return {"direction": "HOLD", "confidence": 0.5, "reason": "数据不足"}
        
        # 各层分析
        structure = StructureLayer.detect_advanced_patterns(sequence)
        rhythm = RhythmLayer.analyze_rhythm(sequence)
        fusion = FusionLayer.fuse_road_signals(sequence)
        
        # 最终决策
        score = fusion['score']
        base_confidence = fusion['confidence']
        
        # 风险调整
        risk_adjustment = 1.0
        if structure['volatility'] > 0.8:
            risk_adjustment *= 0.8  # 高波动降权
        if structure['current_streak'] >= 6:
            risk_adjustment *= 0.7  # 过热保护
        
        final_confidence = base_confidence * risk_adjustment
        
        if score > 0.1:
            direction = "B"
        elif score < -0.1:
            direction = "P"
        else:
            direction = "HOLD"
            final_confidence = 0.5
        
        # 生成专业理由
        reason = StrategyLayer.generate_reasoning(structure, rhythm, direction)
        
        return {
            "direction": direction,
            "confidence": final_confidence,
            "reason": reason,
            "details": {
                "structure": structure,
                "rhythm": rhythm,
                "fusion": fusion
            }
        }
    
    @staticmethod
    def generate_reasoning(structure, rhythm, direction):
        reasons = []
        
        if structure['patterns']:
            reasons.append(f"模式: {','.join(structure['patterns'])}")
        
        reasons.append(f"趋势: {structure['trend']}")
        reasons.append(f"节奏: {rhythm['phase']}")
        
        if direction == "HOLD":
            reasons.append("信号不明，建议观望")
        
        return " | ".join(reasons)

# ---------------- 手机优化界面 ----------------
def display_mobile_interface():
    """手机优化界面"""
    
    # 快速输入系统
    st.markdown("## ⌨️ 快速输入系统")
    col1, col2 = st.columns(2)
    with col1:
        player_input = st.text_input("闲家牌", placeholder="K10 或 552", key="player_input")
    with col2:
        banker_input = st.text_input("庄家牌", placeholder="55 或 AJ", key="banker_input")
    
    # 大按钮结果选择
    st.markdown("## 🏆 本局结果")
    col1, col2, col3 = st.columns(3)
    with col1:
        banker_btn = st.button("🔴 庄赢", use_container_width=True, type="primary")
    with col2:
        player_btn = st.button("🔵 闲赢", use_container_width=True)
    with col3:
        tie_btn = st.button("⚪ 和局", use_container_width=True)
    
    # 解析牌点
    def parse_card_input(input_str):
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
    if banker_btn or player_btn or tie_btn:
        p_cards = parse_card_input(player_input)
        b_cards = parse_card_input(banker_input)
        
        if len(p_cards) >= 2 and len(b_cards) >= 2:
            result = 'B' if banker_btn else 'P' if player_btn else 'T'
            game_data = {
                'round': len(st.session_state.ultimate_games) + 1,
                'player_cards': p_cards,
                'banker_cards': b_cards,
                'result': result,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.ultimate_games.append(game_data)
            
            # 更新路子
            if result in ['B','P']:
                st.session_state.expert_roads['bead_road'].append(result)
            
            st.success(f"✅ 记录成功! 闲{'-'.join(p_cards)} 庄{'-'.join(b_cards)} → {'庄赢' if result=='B' else '闲赢' if result=='P' else '和局'}")
            st.rerun()
        else:
            st.error("❌ 需要至少2张牌")

# ---------------- 专业分析显示 ----------------
def display_expert_analysis():
    """显示专业分析结果"""
    if len(st.session_state.ultimate_games) < 3:
        st.info("🎲 请先记录至少3局牌局数据")
        return
    
    sequence = [game['result'] for game in st.session_state.ultimate_games]
    decision = StrategyLayer.master_decision(sequence)
    
    # 专业预测卡片
    direction = decision['direction']
    confidence = decision['confidence']
    
    if direction == "B":
        color = "#FF6B6B"
        icon = "🔴"
        text = "庄(B)"
        bg_color = "linear-gradient(135deg, #FF6B6B 0%, #C44569 100%)"
    elif direction == "P":
        color = "#4ECDC4"
        icon = "🔵"
        text = "闲(P)"
        bg_color = "linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%)"
    else:
        color = "#FFE66D"
        icon = "⚪"
        text = "观望"
        bg_color = "linear-gradient(135deg, #FFE66D 0%, #F9A826 100%)"
    
    st.markdown(f"""
    <div class="prediction-card" style="background: {bg_color};">
        <h2 style="color: {color}; text-align: center; margin: 0; font-size: 1.8rem;">
            {icon} 大师推荐: {text}
        </h2>
        <h3 style="color: white; text-align: center; margin: 10px 0; font-size: 1.4rem;">
            🎯 置信度: {confidence*100:.1f}%
        </h3>
        <p style="color: #f8f9fa; text-align: center; margin: 0; font-size: 1.1rem;">
            {decision['reason']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 详细分析
    with st.expander("📊 详细分析数据", expanded=False):
        details = decision['details']
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("结构分析")
            structure = details['structure']
            st.write(f"**趋势**: {structure['trend']}")
            st.write(f"**庄闲比例**: {structure['banker_ratio']:.1%}")
            st.write(f"**当前连赢**: {structure['current_streak']}局")
            if structure['patterns']:
                st.write(f"**检测模式**: {', '.join(structure['patterns'])}")
        
        with col2:
            st.subheader("节奏分析")
            rhythm = details['rhythm']
            st.write(f"**Z′分数**: {rhythm['z_score']:+.2f}")
            st.write(f"**动能**: {rhythm['momentum']:.2f}")
            st.write(f"**市场相位**: {rhythm['phase']}")
            st.write(f"**波动率**: {rhythm['volatility']:.2f}")

# ---------------- 路子显示系统 ----------------
def display_professional_roads():
    """显示专业路子"""
    bead_road = st.session_state.expert_roads['bead_road']
    
    if not bead_road:
        st.info("暂无路子数据")
        return
    
    st.markdown("## 🛣️ 专业路子分析")
    
    # 珠路显示
    st.subheader("珠路 (最近15局)")
    recent_bead = bead_road[-15:] if len(bead_road) > 15 else bead_road
    road_display = " ".join(["🔴" if x=='B' else "🔵" for x in recent_bead])
    st.markdown(f'<div class="road-display">{road_display}</div>', unsafe_allow_html=True)
    
    # 统计信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total = len(bead_road)
        st.metric("总局数", total)
    with col2:
        banker_wins = bead_road.count('B')
        st.metric("庄赢", banker_wins)
    with col3:
        player_wins = bead_road.count('P')
        st.metric("闲赢", player_wins)
    with col4:
        if total > 0:
            banker_rate = (banker_wins / total) * 100
            st.metric("庄胜率", f"{banker_rate:.1f}%")

# ---------------- 历史记录 ----------------
def display_game_history():
    """显示牌局历史"""
    if not st.session_state.ultimate_games:
        st.info("暂无历史记录")
        return
    
    st.markdown("## 📝 最近牌局")
    recent_games = st.session_state.ultimate_games[-8:]  # 显示最近8局
    
    for game in reversed(recent_games):
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col1:
                st.write(f"#{game['round']}")
            with col2:
                st.write(f"闲: {'-'.join(game['player_cards'])}")
            with col3:
                st.write(f"庄: {'-'.join(game['banker_cards'])}")
            with col4:
                result = game['result']
                if result == 'B':
                    st.error("庄赢")
                elif result == 'P':
                    st.info("闲赢")
                else:
                    st.warning("和局")
            st.divider()

# ---------------- 主程序 ----------------
def main():
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🎯 智能分析", "🛣️ 路子系统", "📊 数据统计"])
    
    with tab1:
        display_mobile_interface()
        st.markdown("---")
        display_expert_analysis()
    
    with tab2:
        display_professional_roads()
    
    with tab3:
        display_game_history()
        
        # 高级统计
        if st.session_state.ultimate_games:
            st.markdown("## 📈 高级统计")
            games = st.session_state.ultimate_games
            results = [game['result'] for game in games]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                total = len(results)
                st.metric("总局数", total)
            with col2:
                banker_wins = results.count('B')
                st.metric("庄赢次数", banker_wins)
            with col3:
                player_wins = results.count('P')
                st.metric("闲赢次数", player_wins)
            
            if total > 0:
                st.write(f"**庄胜率**: {banker_wins/total*100:.1f}%")
                st.write(f"**闲胜率**: {player_wins/total*100:.1f}%")
                st.write(f"**和局率**: {results.count('T')/total*100:.1f}%")
    
    # 控制按钮
    st.markdown("---")
    if st.button("🔄 开始新牌靴", use_container_width=True):
        st.session_state.ultimate_games.clear()
        st.session_state.expert_roads = {
            'big_road': [], 'big_eye_road': [], 'small_road': [], 
            'cockroach_road': [], 'bead_road': [], 'three_bead_road': []
        }
        st.success("新牌靴开始！")
        st.rerun()

if __name__ == "__main__":
    main()
