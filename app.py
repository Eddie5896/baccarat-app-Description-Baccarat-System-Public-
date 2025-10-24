# -*- coding: utf-8 -*-
# Baccarat Master Ultimate - Precision 12 终极版（含：🛣️看路推荐条）
# 说明：
# 1) 在你的“完全修复版 + 牌点增强系统”基础上，仅新增一个“看路推荐”显示层；
# 2) 不改动你的核心逻辑（六路、60+模式、风控、分析引擎等保持一致）；
# 3) “看路推荐条”显示在智能分析卡上方；纯展示，不影响方向与置信度计算。

import streamlit as st
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
from itertools import groupby

st.set_page_config(page_title="百家乐大师终极版", layout="centered")

# ------------------------ 样式 ------------------------
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
    .multi-road {
        background: #2d3748;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        font-family: monospace;
    }
    .risk-panel {
        background: #2d3748;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #e74c3c;
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
        display: inline-block;
    }
    .road-badge {
        background: #3498db;
        color: white;
        padding: 2px 6px;
        border-radius: 8px;
        font-size: 10px;
        margin: 1px;
    }
    .enhancement-panel {
        background: #2d3748;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #00D4AA;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">🐉 百家乐大师终极版</h1>', unsafe_allow_html=True)

# ------------------------ 状态 ------------------------
if "ultimate_games" not in st.session_state:
    st.session_state.ultimate_games = []
if "expert_roads" not in st.session_state:
    st.session_state.expert_roads = {
        'big_road': [],
        'bead_road': [],
        'big_eye_road': [],
        'small_road': [],
        'cockroach_road': [],
        'three_bead_road': []
    }
if "risk_data" not in st.session_state:
    st.session_state.risk_data = {
        'current_level': 'medium',
        'position_size': 1.0,
        'stop_loss': 3,
        'consecutive_losses': 0,
        'win_streak': 0
    }

# ------------------------ 六路分析 ------------------------
class CompleteRoadAnalyzer:
    """完整六路分析系统"""
    @staticmethod
    def update_all_roads(result):
        if result not in ['B', 'P']:
            return
        roads = st.session_state.expert_roads

        # 1. 珠路
        roads['bead_road'].append(result)

        # 2. 大路
        if not roads['big_road']:
            roads['big_road'].append([result])
        else:
            last_col = roads['big_road'][-1]
            if last_col[-1] == result:
                last_col.append(result)
            else:
                roads['big_road'].append([result])

        # 3. 大眼路（简化一致性比较）
        if len(roads['big_road']) >= 2:
            big_eye = []
            for i in range(1, len(roads['big_road'])):
                if len(roads['big_road'][i]) >= len(roads['big_road'][i-1]):
                    big_eye.append('R')  # 红
                else:
                    big_eye.append('B')  # 蓝
            roads['big_eye_road'] = big_eye[-20:]

        # 4. 小路（大眼路衍生）
        if len(roads['big_eye_road']) >= 2:
            small_road = []
            for i in range(1, len(roads['big_eye_road'])):
                small_road.append('R' if roads['big_eye_road'][i] == roads['big_eye_road'][i-1] else 'B')
            roads['small_road'] = small_road[-15:]

        # 5. 蟑螂路（小路衍生）
        if len(roads['small_road']) >= 2:
            cockroach = []
            for i in range(1, len(roads['small_road'])):
                cockroach.append('R' if roads['small_road'][i] == roads['small_road'][i-1] else 'B')
            roads['cockroach_road'] = cockroach[-12:]

        # 6. 三珠路
        bead_road = roads['bead_road']
        if len(bead_road) >= 3:
            groups = [bead_road[i:i+3] for i in range(0, len(bead_road)-2, 3)]
            roads['three_bead_road'] = groups[-8:]

# ------------------------ 模式检测（含 60+ 类） ------------------------
class AdvancedPatternDetector:
    @staticmethod
    def detect_all_patterns(sequence):
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 4: return []
        patterns = []
        try:
            patterns.extend(AdvancedPatternDetector.detect_dragon_patterns(bp_seq))
            patterns.extend(AdvancedPatternDetector.detect_jump_patterns(bp_seq))
            patterns.extend(AdvancedPatternDetector.detect_house_patterns(bp_seq))
            patterns.extend(AdvancedPatternDetector.detect_trend_patterns(bp_seq))
            patterns.extend(AdvancedPatternDetector.detect_road_patterns(bp_seq))
            patterns.extend(AdvancedPatternDetector.detect_special_patterns(bp_seq))
            patterns.extend(AdvancedPatternDetector.detect_water_patterns(bp_seq))
            patterns.extend(AdvancedPatternDetector.detect_graph_patterns(bp_seq))
        except Exception:
            patterns.extend(AdvancedPatternDetector.detect_basic_patterns(bp_seq))
        return patterns[:8]

    @staticmethod
    def detect_basic_patterns(bp_seq):
        patterns = []
        if len(bp_seq) >= 4:
            last_4 = bp_seq[-4:]
            if len(set(last_4)) == 1:
                patterns.append(f"{bp_seq[-1]}长龙")
        return patterns

    @staticmethod
    def get_streaks(bp_seq):
        if not bp_seq: return []
        streaks, current, count = [], bp_seq[0], 1
        for i in range(1, len(bp_seq)):
            if bp_seq[i] == current: count += 1
            else:
                streaks.append(count); current = bp_seq[i]; count = 1
        streaks.append(count)
        return streaks

    # --- 龙系列 ---
    @staticmethod
    def detect_dragon_patterns(bp_seq):
        patterns = []
        if len(bp_seq) < 4: return patterns
        last_4 = bp_seq[-4:]
        if len(set(last_4)) == 1: patterns.append(f"{bp_seq[-1]}长龙")
        if len(bp_seq) >= 5 and len(set(bp_seq[-5:])) == 1: patterns.append(f"强{bp_seq[-1]}长龙")
        if len(bp_seq) >= 6 and len(set(bp_seq[-6:])) == 1: patterns.append(f"超强{bp_seq[-1]}长龙")
        return patterns

    # --- 跳系列 ---
    @staticmethod
    def detect_jump_patterns(bp_seq):
        patterns = []
        if len(bp_seq) < 6: return patterns
        last_6 = bp_seq[-6:]
        if last_6 in [['B','P','B','P','B','P'], ['P','B','P','B','P','B']]: patterns.append("完美单跳")
        if len(bp_seq) >= 8:
            last_8 = bp_seq[-8:]
            if last_8 in [['B','B','P','P','B','B','P','P'], ['P','P','B','B','P','P','B','B']]:
                patterns.append("齐头双跳")
        if len(bp_seq) >= 5:
            last_5 = bp_seq[-5:]
            if last_5 in [['B','P','B','P','B'], ['P','B','P','B','P']]:
                patterns.append("长短单跳")
        return patterns

    # --- 房厅系列 ---
    @staticmethod
    def detect_house_patterns(bp_seq):
        patterns = []
        if len(bp_seq) < 5: return patterns
        streaks = AdvancedPatternDetector.get_streaks(bp_seq)
        if len(streaks) < 3: return patterns
        try:
            if len(streaks) >= 3 and (streaks[-3] == 2 and streaks[-2] == 1 and streaks[-1] == 2):
                patterns.append("一房一厅")
            if len(streaks) >= 4 and (streaks[-4] == 2 and streaks[-3] == 2 and streaks[-2] == 1 and streaks[-1] == 2):
                patterns.append("两房一厅")
            if len(streaks) >= 4 and (streaks[-4] >= 3 and streaks[-3] >= 3 and streaks[-2] == 1 and streaks[-1] >= 3):
                patterns.append("三房一厅")
            if len(streaks) >= 4 and (streaks[-4] >= 4 and streaks[-3] >= 4 and streaks[-2] == 1 and streaks[-1] >= 4):
                patterns.append("四房一厅")
            if len(streaks) >= 4 and (streaks[-4] >= 3 and streaks[-3] >= 3 and streaks[-2] == 1 and streaks[-1] == 2):
                patterns.append("假三房")
        except Exception:
            pass
        return patterns

    # --- 趋势系列 ---
    @staticmethod
    def detect_trend_patterns(bp_seq):
        patterns = []
        if len(bp_seq) < 6: return patterns
        try:
            streaks = AdvancedPatternDetector.get_streaks(bp_seq)
            if len(streaks) < 4: return patterns
            if len(streaks) >= 4 and all(streaks[i] < streaks[i+1] for i in range(-4, -1)):
                patterns.append("上山路")
            if len(streaks) >= 4 and all(streaks[i] > streaks[i+1] for i in range(-4, -1)):
                patterns.append("下山路")
            if len(streaks) >= 5:
                if (streaks[-5] < streaks[-4] > streaks[-3] < streaks[-2] > streaks[-1] or
                    streaks[-5] > streaks[-4] < streaks[-3] > streaks[-2] < streaks[-1]):
                    patterns.append("楼梯路")
        except Exception:
            pass
        return patterns

    # --- 水路 ---
    @staticmethod
    def detect_water_patterns(bp_seq):
        patterns = []
        if len(bp_seq) < 8: return patterns
        try:
            changes = sum(1 for i in range(1, len(bp_seq)) if bp_seq[i] != bp_seq[i-1])
            volatility = changes / len(bp_seq)
            if volatility < 0.3: patterns.append("静水路")
            elif volatility < 0.6: patterns.append("微澜路")
            else: patterns.append("激流路")
        except Exception:
            pass
        return patterns

    # --- 特殊格局 ---
    @staticmethod
    def detect_special_patterns(bp_seq):
        patterns = []
        if len(bp_seq) < 5: return patterns
        try:
            streaks = AdvancedPatternDetector.get_streaks(bp_seq)
            if len(streaks) >= 3 and (streaks[-3] >= 3 and streaks[-2] == 1 and streaks[-1] >= 3):
                patterns.append("回头龙")
            b_ratio = bp_seq.count('B') / len(bp_seq)
            if b_ratio > 0.7: patterns.append("庄王格局")
            elif b_ratio < 0.3: patterns.append("闲霸格局")
            elif 0.45 <= b_ratio <= 0.55: patterns.append("平衡格局")
        except Exception:
            pass
        return patterns

    # --- 预留扩展 ---
    @staticmethod
    def detect_road_patterns(bp_seq): return []
    @staticmethod
    def detect_graph_patterns(bp_seq): return []

# ------------------------ 牌点增强分析（保持原逻辑） ------------------------
class CardEnhancementAnalyzer:
    @staticmethod
    def analyze_card_enhancement(games_with_cards):
        if len(games_with_cards) < 3:
            return {"enhancement_factor": 0, "reason": "数据不足"}
        card_games = [g for g in games_with_cards if g.get('mode') == 'card'
                      and len(g['player_cards']) >= 2 and len(g['banker_cards']) >= 2]
        if len(card_games) < 2:
            return {"enhancement_factor": 0, "reason": "牌点数据不足"}

        enhancement, reasons = 0, []
        try:
            nat = CardEnhancementAnalyzer._analyze_natural_effect(card_games)
            if nat['factor'] != 0: enhancement += nat['factor']; reasons.append(nat['reason'])
            mom = CardEnhancementAnalyzer._analyze_point_momentum(card_games)
            if mom['factor'] != 0: enhancement += mom['factor']; reasons.append(mom['reason'])
            draw = CardEnhancementAnalyzer._analyze_draw_patterns(card_games)
            if draw['factor'] != 0: enhancement += draw['factor']; reasons.append(draw['reason'])
        except Exception:
            return {"enhancement_factor": 0, "reason": "分析异常"}

        return {
            "enhancement_factor": max(-0.2, min(0.2, enhancement)),
            "reason": " | ".join(reasons) if reasons else "无增强信号"
        }

    @staticmethod
    def _analyze_natural_effect(card_games):
        if len(card_games) < 3: return {"factor": 0, "reason": ""}
        recent = card_games[-3:]; cnt = 0
        for g in recent:
            p = CardEnhancementAnalyzer._calculate_points(g['player_cards'])
            b = CardEnhancementAnalyzer._calculate_points(g['banker_cards'])
            if p >= 8 or b >= 8: cnt += 1
        if cnt >= 2: return {"factor": 0.08, "reason": f"天牌密集({cnt}局)"}
        if cnt == 1: return {"factor": 0.03, "reason": "天牌出现"}
        return {"factor": 0, "reason": ""}

    @staticmethod
    def _analyze_point_momentum(card_games):
        if len(card_games) < 4: return {"factor": 0, "reason": ""}
        pts = []
        for g in card_games[-4:]:
            pts.append(CardEnhancementAnalyzer._calculate_points(g['player_cards']))
            pts.append(CardEnhancementAnalyzer._calculate_points(g['banker_cards']))
        avg_pt = sum(pts)/len(pts)
        if avg_pt < 4: return {"factor": 0.06, "reason": "小点数期"}
        if avg_pt > 7: return {"factor": -0.04, "reason": "大点数期"}
        return {"factor": 0, "reason": ""}

    @staticmethod
    def _analyze_draw_patterns(card_games):
        if len(card_games) < 5: return {"factor": 0, "reason": ""}
        draw_count = 0
        total = min(10, len(card_games))
        for g in card_games[-total:]:
            p = CardEnhancementAnalyzer._calculate_points(g['player_cards'])
            b = CardEnhancementAnalyzer._calculate_points(g['banker_cards'])
            if p < 6 or b < 6: draw_count += 1
        ratio = draw_count/total
        if ratio > 0.7: return {"factor": -0.05, "reason": "补牌密集"}
        if ratio < 0.3: return {"factor": 0.04, "reason": "补牌稀少"}
        return {"factor": 0, "reason": ""}

    @staticmethod
    def _calculate_points(cards):
        mp = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':0,'J':0,'Q':0,'K':0}
        total = sum(mp.get(c,0) for c in cards)
        return total % 10

# ------------------------ 风控 ------------------------
class ProfessionalRiskManager:
    @staticmethod
    def calculate_position_size(confidence, streak_info):
        base = 1.0
        if confidence > 0.8: base *= 1.2
        elif confidence > 0.7: base *= 1.0
        elif confidence > 0.6: base *= 0.8
        else: base *= 0.5
        if streak_info.get('current_streak',0) >= 3: base *= 1.1
        if st.session_state.risk_data['consecutive_losses'] >= 2: base *= 0.7
        elif st.session_state.risk_data['consecutive_losses'] >= 3: base *= 0.5
        return min(base, 2.0)

    @staticmethod
    def get_risk_level(confidence, volatility):
        risk_score = (1 - confidence) + volatility
        if risk_score < 0.3: return "low", "🟢 低风险"
        if risk_score < 0.6: return "medium", "🟡 中风险"
        if risk_score < 0.8: return "high", "🟠 高风险"
        return "extreme", "🔴 极高风险"

    @staticmethod
    def get_trading_suggestion(risk_level, direction):
        s = {
            "low":{"B":"✅ 庄势明确，可适度加仓","P":"✅ 闲势明确，可适度加仓","HOLD":"⚪ 趋势平衡，正常操作"},
            "medium":{"B":"⚠️ 庄势一般，建议轻仓","P":"⚠️ 闲势一般，建议轻仓","HOLD":"⚪ 信号不明，建议观望"},
            "high":{"B":"🚨 高波动庄势，谨慎操作","P":"🚨 高波动闲势，谨慎操作","HOLD":"⛔ 高风险期，建议休息"},
            "extreme":{"B":"⛔ 极高风险，强烈建议观望","P":"⛔ 极高风险，强烈建议观望","HOLD":"⛔ 市场混乱，暂停交易"}
        }
        return s[risk_level].get(direction,"正常操作")

# ------------------------ 分析引擎（保持原逻辑） ------------------------
class UltimateAnalysisEngine:
    @staticmethod
    def comprehensive_analysis(sequence):
        if len(sequence) < 4:
            return {"direction":"HOLD","confidence":0.5,"reason":"数据不足，请记录更多牌局","patterns":[],"risk_level":"medium","risk_text":"🟡 中风险","current_streak":0,"volatility":0}
        bp_seq = [x for x in sequence if x in ['B','P']]

        # 1) 模式
        patterns = AdvancedPatternDetector.detect_all_patterns(sequence)
        current_streak = UltimateAnalysisEngine.get_current_streak(bp_seq)

        # 2) 趋势
        b_ratio = bp_seq.count('B')/len(bp_seq) if bp_seq else 0.5
        recent_8 = bp_seq[-8:] if len(bp_seq) >= 8 else bp_seq
        b_recent = recent_8.count('B')/len(recent_8) if recent_8 else 0.5

        # 3) 动能
        volatility = UltimateAnalysisEngine.calculate_volatility(bp_seq)
        momentum = UltimateAnalysisEngine.calculate_momentum(bp_seq)

        # 4) 决策融合
        base = 0
        if patterns: base += len(patterns)*0.1
        if b_ratio > 0.6: base += 0.3
        elif b_ratio < 0.4: base -= 0.3
        if b_recent > 0.75: base += 0.2
        elif b_recent < 0.25: base -= 0.2
        if current_streak >= 3:
            d = bp_seq[-1]
            base += current_streak*0.1 if d == "B" else -current_streak*0.1
        base += momentum*0.2

        confidence = 0.5
        confidence += abs(base)*0.4
        confidence += len(patterns)*0.1
        confidence = min(confidence, 0.9)

        if base > 0.15: direction = "B"
        elif base < -0.15: direction = "P"
        else: direction, confidence = "HOLD", 0.5

        risk_level, risk_text = ProfessionalRiskManager.get_risk_level(confidence, volatility)
        reason = UltimateAnalysisEngine.generate_reasoning(patterns, direction, current_streak, risk_level)

        return {"direction":direction,"confidence":confidence,"reason":reason,"patterns":patterns,
                "risk_level":risk_level,"risk_text":risk_text,"current_streak":current_streak,"volatility":volatility}

    @staticmethod
    def get_current_streak(bp_seq):
        if not bp_seq: return 0
        cur, streak = bp_seq[-1], 1
        for i in range(len(bp_seq)-2, -1, -1):
            if bp_seq[i] == cur: streak += 1
            else: break
        return streak

    @staticmethod
    def calculate_volatility(bp_seq):
        if len(bp_seq) < 2: return 0
        changes = sum(1 for i in range(1, len(bp_seq)) if bp_seq[i] != bp_seq[i-1])
        return changes / len(bp_seq)

    @staticmethod
    def calculate_momentum(bp_seq):
        if len(bp_seq) < 4: return 0
        recent = bp_seq[-4:]
        return sum(1 for x in recent if x == recent[-1]) / len(recent) - 0.5

    @staticmethod
    def generate_reasoning(patterns, direction, streak, risk_level):
        reasons = []
        if patterns: reasons.append(f"模式:{','.join(patterns[:3])}")
        if streak >= 2: reasons.append(f"连{streak}局")
        reasons.append(f"风险:{risk_level}")
        if direction == "HOLD": reasons.append("建议观望")
        return " | ".join(reasons)

# ------------------------ 新增：看路推荐（纯显示层） ------------------------
def road_recommendation(roads):
    """
    传统看路推荐（不影响主引擎）：
    - 以大路为主，小路/大眼/蟑螂为辅；
    - 返回 {'lines':[...], 'final':'xxx'}
    """
    lines = []
    final = ""

    # 大路：主导
    if roads['big_road']:
        last_col = roads['big_road'][-1]
        color_cn = "庄" if last_col[-1] == 'B' else "闲"
        streak = len(last_col)
        if streak >= 3:
            lines.append(f"大路：{color_cn}连{streak}局 → 顺路{color_cn}")
            final = f"顺大路{color_cn}"
        else:
            lines.append(f"大路：{color_cn}走势平衡")

    # 大眼路：稳定度
    if roads['big_eye_road']:
        r = roads['big_eye_road'].count('R')
        b = roads['big_eye_road'].count('B')
        if r > b: lines.append("大眼路：红>蓝 → 趋势延续")
        elif b > r: lines.append("大眼路：蓝>红 → 有反转迹象")
        else: lines.append("大眼路：红=蓝 → 稳定期")

    # 小路：节奏
    if roads['small_road']:
        r = roads['small_road'].count('R')
        b = roads['small_road'].count('B')
        if r > b: lines.append("小路：红>蓝 → 延续趋势")
        elif b > r: lines.append("小路：蓝>红 → 节奏转弱")
        else: lines.append("小路：红=蓝 → 平衡")

    # 蟑螂路：短期震荡
    if roads['cockroach_road']:
        last3 = roads['cockroach_road'][-3:]
        if not last3:
            pass
        else:
            trend = "红红蓝" if last3.count('R') == 2 else ("蓝蓝红" if last3.count('B') == 2 else "混乱")
            lines.append(f"蟑螂路：{trend} → {'轻微震荡' if trend!='混乱' else '趋势不明'}")

    if not final:
        # 若大路没给出明确顺路，则基于辅路给一个温和建议
        if roads['big_eye_road']:
            r = roads['big_eye_road'].count('R'); b = roads['big_eye_road'].count('B')
            if r > b: final = "顺路（偏红，延续）"
            elif b > r: final = "反路（偏蓝，注意反转）"
            else: final = "暂无明显方向"
        else:
            final = "暂无明显方向"

    return {"lines": lines, "final": final}

# ------------------------ 输入界面 ------------------------
def display_complete_interface():
    st.markdown("## 🎮 双模式输入系统")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🃏 牌点输入", use_container_width=True, type="primary"):
            st.session_state.input_mode = "card"; st.rerun()
    with col2:
        if st.button("🎯 快速看路", use_container_width=True):
            st.session_state.input_mode = "result"; st.rerun()
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "card"
    if st.session_state.input_mode == "card":
        display_card_input()
    else:
        display_quick_input()

def display_card_input():
    col1, col2 = st.columns(2)
    with col1:
        player_input = st.text_input("闲家牌", placeholder="K10 或 552", key="player_card")
    with col2:
        banker_input = st.text_input("庄家牌", placeholder="55 或 AJ", key="banker_card")

    st.markdown("### 🏆 本局结果")
    c1, c2, c3 = st.columns(3)
    with c1: banker_btn = st.button("🔴 庄赢", use_container_width=True, type="primary")
    with c2: player_btn = st.button("🔵 闲赢", use_container_width=True)
    with c3: tie_btn = st.button("⚪ 和局", use_container_width=True)
    if banker_btn or player_btn or tie_btn:
        handle_card_input(player_input, banker_input, banker_btn, player_btn, tie_btn)

def display_quick_input():
    st.info("💡 快速模式：直接记录结果，用于快速看路分析")
    c1, c2 = st.columns(2)
    with c1: quick_banker = st.button("🔴 庄赢", use_container_width=True, type="primary")
    with c2: quick_player = st.button("🔵 闲赢", use_container_width=True)
    st.markdown("### 📝 批量输入")
    batch_input = st.text_input("输入BP序列", placeholder="BPBBP 或 庄闲庄庄闲", key="batch_input")
    if st.button("✅ 确认批量输入", use_container_width=True) and batch_input:
        handle_batch_input(batch_input)
    if quick_banker or quick_player:
        handle_quick_input(quick_banker, quick_player)

def parse_cards(input_str):
    if not input_str: return []
    s = input_str.upper().replace(' ',''); cards=[]; i=0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2]=='10': cards.append('10'); i+=2
        elif s[i] in '123456789': cards.append(s[i]); i+=1
        elif s[i] in ['A','J','Q','K','0']:
            card_map={'A':'A','J':'J','Q':'Q','K':'K','0':'10'}
            cards.append(card_map[s[i]]); i+=1
        else: i+=1
    return cards

def handle_card_input(player_input, banker_input, banker_btn, player_btn, tie_btn):
    p_cards = parse_cards(player_input)
    b_cards = parse_cards(banker_input)
    if len(p_cards) >= 2 and len(b_cards) >= 2:
        result = 'B' if banker_btn else ('P' if player_btn else 'T')
        record_game(result, p_cards, b_cards, 'card')
    else:
        st.error("❌ 需要至少2张牌")

def handle_quick_input(quick_banker, quick_player):
    result = 'B' if quick_banker else 'P'
    record_game(result, ['X','X'], ['X','X'], 'quick')

def handle_batch_input(batch_input):
    s = batch_input.upper().replace('庄','B').replace('闲','P').replace(' ','')
    valid = [c for c in s if c in ['B','P']]
    if valid:
        for r in valid:
            record_game(r, ['X','X'], ['X','X'], 'batch')
        st.success(f"✅ 批量添加{len(valid)}局")

def record_game(result, p_cards, b_cards, mode):
    game = {
        'round': len(st.session_state.ultimate_games) + 1,
        'player_cards': p_cards,
        'banker_cards': b_cards,
        'result': result,
        'time': datetime.now().strftime("%H:%M"),
        'mode': mode
    }
    st.session_state.ultimate_games.append(game)
    if result in ['B','P']:
        CompleteRoadAnalyzer.update_all_roads(result)
    update_risk_data(result)
    st.success(f"✅ 记录成功! 第{game['round']}局")
    st.rerun()

def update_risk_data(result):
    risk = st.session_state.risk_data
    if result in ['B','P']:
        risk['win_streak'] += 1
        risk['consecutive_losses'] = 0
    else:
        risk['consecutive_losses'] += 1
        risk['win_streak'] = 0

# ------------------------ 展示：智能分析 + 看路推荐条 ------------------------
def display_complete_analysis():
    if len(st.session_state.ultimate_games) < 3:
        st.info("🎲 请先记录至少3局牌局数据"); return

    sequence = [g['result'] for g in st.session_state.ultimate_games]

    # 原有分析（保留原逻辑）
    analysis = UltimateAnalysisEngine.comprehensive_analysis(sequence)

    # ========= 新增：看路推荐条（显示在分析卡之上） =========
    road_sug = road_recommendation(st.session_state.expert_roads)
    if road_sug and road_sug.get("final"):
        st.markdown(f"""
        <div style="
            background: linear-gradient(90deg, #FFD70033, #FF634733);
            padding: 10px 14px;
            border-radius: 10px;
            margin-top: 6px; margin-bottom: 10px;
            border-left: 5px solid #FFD700;
            color: #FFFFFF;
            font-weight: 600;
            text-shadow: 1px 1px 2px #000;">
            🛣️ 看路推荐：{road_sug['final']}
        </div>
        """, unsafe_allow_html=True)

    # ======= 原有预测卡片（未改动） =======
    direction = analysis['direction']; confidence = analysis['confidence']
    reason = analysis['reason']; patterns = analysis.get('patterns', [])
    risk_level = analysis.get('risk_level','medium'); risk_text = analysis.get('risk_text','🟡 中风险')

    if direction == "B":
        color="#FF6B6B"; icon="🔴"; text="庄(B)"; bg="linear-gradient(135deg, #FF6B6B 0%, #C44569 100%)"
    elif direction == "P":
        color="#4ECDC4"; icon="🔵"; text="闲(P)"; bg="linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%)"
    else:
        color="#FFE66D"; icon="⚪"; text="观望"; bg="linear-gradient(135deg, #FFE66D 0%, #F9A826 100%)"

    st.markdown(f"""
    <div class="prediction-card" style="background: {bg};">
        <h2 style="color: {color}; text-align: center; margin: 0;">
            {icon} 大师推荐: {text}
        </h2>
        <h3 style="color: white; text-align: center; margin: 10px 0;">
            🎯 置信度: {confidence*100:.1f}% | {risk_text}
        </h3>
        <p style="color: #f8f9fa; text-align: center; margin: 0;">
            {reason}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 模式显示
    if patterns:
        st.markdown("### 🧩 检测模式")
        html = "".join([f'<span class="pattern-badge">{p}</span>' for p in patterns[:5]])
        st.markdown(html, unsafe_allow_html=True)

    # 风险控制
    display_risk_panel(analysis)

def display_risk_panel(analysis):
    st.markdown("### 🛡️ 风险控制")
    position_size = ProfessionalRiskManager.calculate_position_size(
        analysis['confidence'], {'current_streak': analysis.get('current_streak',0)}
    )
    suggestion = ProfessionalRiskManager.get_trading_suggestion(analysis['risk_level'], analysis['direction'])
    st.markdown(f"""
    <div class="risk-panel">
        <h4 style="color: white; margin: 0 0 10px 0;">📊 风险控制建议</h4>
        <p style="color: #ccc; margin: 5px 0;"><strong>仓位建议:</strong> {position_size:.1f}倍基础仓位</p>
        <p style="color: #ccc; margin: 5px 0;"><strong>操作建议:</strong> {suggestion}</p>
        <p style="color: #ccc; margin: 5px 0;"><strong>连赢:</strong> {st.session_state.risk_data['win_streak']}局 | <strong>连输:</strong> {st.session_state.risk_data['consecutive_losses']}局</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------ 六路展示 ------------------------
def display_complete_roads():
    roads = st.session_state.expert_roads
    st.markdown("## 🛣️ 完整六路分析")

    st.markdown("#### 🟠 珠路 (最近20局)")
    if roads['bead_road']:
        bead_display = " ".join(["🔴" if x=='B' else "🔵" for x in roads['bead_road'][-20:]])
        st.markdown(f'<div class="road-display">{bead_display}</div>', unsafe_allow_html=True)

    st.markdown("#### 🔴 大路")
    if roads['big_road']:
        for i, col in enumerate(roads['big_road'][-6:]):
            col_display = " ".join(["🔴" if x=='B' else "🔵" for x in col])
            st.markdown(f'<div class="multi-road">第{i+1}列: {col_display}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if roads['big_eye_road']:
            st.markdown("#### 👁️ 大眼路")
            display = " ".join(["🔴" if x=='R' else "🔵" for x in roads['big_eye_road'][-12:]])
            st.markdown(f'<div class="multi-road">{display}</div>', unsafe_allow_html=True)
    with col2:
        if roads['small_road']:
            st.markdown("#### 🔵 小路")
            display = " ".join(["🔴" if x=='R' else "🔵" for x in roads['small_road'][-10:]])
            st.markdown(f'<div class="multi-road">{display}</div>', unsafe_allow_html=True)

    if roads['three_bead_road']:
        st.markdown("#### 🔶 三珠路")
        for i, group in enumerate(roads['three_bead_road'][-6:]):
            display = " ".join(["🔴" if x=='B' else "🔵" for x in group])
            st.markdown(f'<div class="multi-road">第{i+1}组: {display}</div>', unsafe_allow_html=True)

# ------------------------ 统计 ------------------------
def display_professional_stats():
    if not st.session_state.ultimate_games:
        st.info("暂无统计数据"); return

    games = st.session_state.ultimate_games
    results = [g['result'] for g in games]
    bead_road = st.session_state.expert_roads['bead_road']

    st.markdown("## 📊 专业统计")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("总局数", len(results))
    with c2: st.metric("庄赢", results.count('B'))
    with c3: st.metric("闲赢", results.count('P'))
    with c4: st.metric("和局", results.count('T'))

    if bead_road:
        st.markdown("#### 📈 高级分析")
        c1,c2,c3 = st.columns(3)
        with c1:
            total = len(results)
            if total>0:
                st.metric("庄胜率", f"{results.count('B')/total*100:.1f}%")
        with c2:
            avg_streak = np.mean([len(list(g)) for k,g in groupby(bead_road)]) if len(bead_road)>0 else 0
            st.metric("平均连赢", f"{avg_streak:.1f}局")
        with c3:
            if len(bead_road)>1:
                changes = sum(1 for i in range(1,len(bead_road)) if bead_road[i]!=bead_road[i-1])
                vol = changes/len(bead_road)*100
                st.metric("波动率", f"{vol:.1f}%")

# ------------------------ 历史 ------------------------
def display_complete_history():
    if not st.session_state.ultimate_games:
        st.info("暂无历史记录"); return
    st.markdown("## 📝 完整历史")
    recent = st.session_state.ultimate_games[-10:]
    for g in reversed(recent):
        icon = "🃏" if g.get('mode')=='card' else ("🎯" if g.get('mode')=='quick' else "📝")
        with st.container():
            c1,c2,c3,c4,c5 = st.columns([1,1,2,2,1])
            with c1: st.write(f"#{g['round']}")
            with c2: st.write(icon)
            with c3: st.write(f"闲: {'-'.join(g['player_cards'])}" if g.get('mode')=='card' else "快速记录")
            with c4: st.write(f"庄: {'-'.join(g['banker_cards'])}" if g.get('mode')=='card' else "快速记录")
            with c5:
                if g['result']=='B': st.error("庄赢")
                elif g['result']=='P': st.info("闲赢")
                else: st.warning("和局")

# ------------------------ 主程序 ------------------------
def main():
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 智能分析", "🛣️ 六路分析", "📊 专业统计", "📝 历史记录"])
    with tab1:
        display_complete_interface()
        st.markdown("---")
        display_complete_analysis()
    with tab2:
        display_complete_roads()
    with tab3:
        display_professional_stats()
    with tab4:
        display_complete_history()

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 开始新牌靴", use_container_width=True):
            st.session_state.ultimate_games.clear()
            st.session_state.expert_roads = {
                'big_road': [], 'bead_road': [], 'big_eye_road': [],
                'small_road': [], 'cockroach_road': [], 'three_bead_road': []
            }
            st.session_state.risk_data = {
                'current_level':'medium','position_size':1.0,
                'stop_loss':3,'consecutive_losses':0,'win_streak':0
            }
            st.success("新牌靴开始！"); st.rerun()
    with c2:
        if st.button("📋 导出数据", use_container_width=True):
            st.info("数据导出功能准备中...")

if __name__ == "__main__":
    main()
