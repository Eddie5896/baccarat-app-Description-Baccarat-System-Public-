# -*- coding: utf-8 -*-
# Baccarat Master - Precision 12 Hybrid + EOR Control Ultimate (Full Route Vision Edition)
# ✅ 全功能：六路 + 60+模式 + 看路推荐 + 状态信号 + Hybrid 数值 + EOR 副数调节 + 风控 + 牌点/快速/批量输入

import streamlit as st
import numpy as np
from datetime import datetime
from collections import Counter
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
    .state-signal {
        background: linear-gradient(90deg, #FFD70033, #FF634733);
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #FFD700;
        color: #FFFFFF;
        font-weight: 600;
    }
    .risk-panel {
        background: #2d3748;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #e74c3c;
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
</style>
""", unsafe_allow_html=True)

# ------------------------ 状态初始化 ------------------------
if "ultimate_games" not in st.session_state:
    st.session_state.ultimate_games = []  # 每局：result, player_cards, banker_cards, mode, time
if "expert_roads" not in st.session_state:
    st.session_state.expert_roads = {
        'big_road': [], 'bead_road': [], 'big_eye_road': [],
        'small_road': [], 'cockroach_road': [], 'three_bead_road': []
    }
if "risk_data" not in st.session_state:
    st.session_state.risk_data = {'consecutive_losses': 0, 'win_streak': 0}
if "eor_decks" not in st.session_state:
    st.session_state["eor_decks"] = 8  # 默认 8 副，允许 0 代表关闭 EOR 增强

# ------------------------ 工具函数 ------------------------
def parse_cards(input_str):
    if not input_str: return []
    s = input_str.upper().replace(' ', '')
    cards, i = [], 0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2] == '10': cards.append('10'); i += 2
        elif s[i] in '123456789': cards.append(s[i]); i += 1
        elif s[i] in ['A','J','Q','K','0']:
            m = {'A':'A','J':'J','Q':'Q','K':'K','0':'10'}
            cards.append(m[s[i]]); i += 1
        else:
            i += 1
    return cards

# ------------------------ 六路分析 ------------------------
class CompleteRoadAnalyzer:
    """完整六路分析系统：珠路/大路/大眼路/小路/蟑螂路/三珠路"""
    @staticmethod
    def update_all_roads(result):
        if result not in ['B', 'P']: return
        R = st.session_state.expert_roads

        # 1) 珠路
        R['bead_road'].append(result)

        # 2) 大路（按列）
        if not R['big_road']:
            R['big_road'].append([result])
        else:
            last_col = R['big_road'][-1]
            if last_col[-1] == result:
                last_col.append(result)
            else:
                R['big_road'].append([result])

        # 3) 大眼路（看大路列长度对比）
        if len(R['big_road']) >= 2:
            big_eye = []
            for i in range(1, len(R['big_road'])):
                big_eye.append('R' if len(R['big_road'][i]) >= len(R['big_road'][i-1]) else 'B')
            R['big_eye_road'] = big_eye[-20:]

        # 4) 小路（大眼路相邻一致/不一致）
        if len(R['big_eye_road']) >= 2:
            small = []
            for i in range(1, len(R['big_eye_road'])):
                small.append('R' if R['big_eye_road'][i] == R['big_eye_road'][i-1] else 'B')
            R['small_road'] = small[-15:]

        # 5) 蟑螂路（小路相邻一致/不一致）
        if len(R['small_road']) >= 2:
            cock = []
            for i in range(1, len(R['small_road'])):
                cock.append('R' if R['small_road'][i] == R['small_road'][i-1] else 'B')
            R['cockroach_road'] = cock[-12:]

        # 6) 三珠路（每三局一组）
        bead = R['bead_road']
        if len(bead) >= 3:
            groups = [bead[i:i+3] for i in range(0, len(bead)-2, 3)]
            R['three_bead_road'] = groups[-8:]

# ------------------------ 60+ 模式检测 ------------------------
class AdvancedPatternDetector:
    @staticmethod
    def get_streaks(bp_seq):
        if not bp_seq: return []
        streaks, cur, cnt = [], bp_seq[0], 1
        for i in range(1, len(bp_seq)):
            if bp_seq[i] == cur: cnt += 1
            else: streaks.append(cnt); cur = bp_seq[i]; cnt = 1
        streaks.append(cnt); return streaks

    @staticmethod
    def detect_all_patterns(sequence):
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 4: return []
        P = []
        try:
            P.extend(AdvancedPatternDetector.detect_dragon_patterns(bp_seq))
            P.extend(AdvancedPatternDetector.detect_jump_patterns(bp_seq))
            P.extend(AdvancedPatternDetector.detect_house_patterns(bp_seq))
            P.extend(AdvancedPatternDetector.detect_trend_patterns(bp_seq))
            P.extend(AdvancedPatternDetector.detect_water_patterns(bp_seq))
            P.extend(AdvancedPatternDetector.detect_special_patterns(bp_seq))
            # detect_road_patterns / detect_graph_patterns 预留
        except Exception:
            if len(set(bp_seq[-4:])) == 1: P.append(f"{bp_seq[-1]}长龙")
        return P[:8]

    # 龙系列
    @staticmethod
    def detect_dragon_patterns(bp_seq):
        P=[]
        if len(set(bp_seq[-4:]))==1: P.append(f"{bp_seq[-1]}长龙")
        if len(bp_seq)>=5 and len(set(bp_seq[-5:]))==1: P.append(f"强{bp_seq[-1]}长龙")
        if len(bp_seq)>=6 and len(set(bp_seq[-6:]))==1: P.append(f"超强{bp_seq[-1]}长龙")
        return P

    # 跳系列
    @staticmethod
    def detect_jump_patterns(bp_seq):
        P=[]
        if len(bp_seq)>=6 and bp_seq[-6:] in [['B','P','B','P','B','P'],['P','B','P','B','P','B']]: P.append("完美单跳")
        if len(bp_seq)>=8 and bp_seq[-8:] in [['B','B','P','P','B','B','P','P'],['P','P','B','B','P','P','B','B']]: P.append("齐头双跳")
        if len(bp_seq)>=5 and bp_seq[-5:] in [['B','P','B','P','B'],['P','B','P','B','P']]: P.append("长短单跳")
        return P

    # 房厅系列
    @staticmethod
    def detect_house_patterns(bp_seq):
        P=[]; S=AdvancedPatternDetector.get_streaks(bp_seq)
        if len(S)<3: return P
        if len(S)>=3 and (S[-3]==2 and S[-2]==1 and S[-1]==2): P.append("一房一厅")
        if len(S)>=4 and (S[-4]==2 and S[-3]==2 and S[-2]==1 and S[-1]==2): P.append("两房一厅")
        if len(S)>=4 and (S[-4]>=3 and S[-3]>=3 and S[-2]==1 and S[-1]>=3): P.append("三房一厅")
        if len(S)>=4 and (S[-4]>=4 and S[-3]>=4 and S[-2]==1 and S[-1]>=4): P.append("四房一厅")
        if len(S)>=4 and (S[-4]>=3 and S[-3]>=3 and S[-2]==1 and S[-1]==2): P.append("假三房")
        return P

    # 趋势系列
    @staticmethod
    def detect_trend_patterns(bp_seq):
        P=[]; S=AdvancedPatternDetector.get_streaks(bp_seq)
        if len(S)<4: return P
        if all(S[i]<S[i+1] for i in range(-4,-1)): P.append("上山路")
        if all(S[i]>S[i+1] for i in range(-4,-1)): P.append("下山路")
        if len(S)>=5 and ((S[-5] < S[-4] > S[-3] < S[-2] > S[-1]) or (S[-5] > S[-4] < S[-3] > S[-2] < S[-1])): P.append("楼梯路")
        return P

    # 水路
    @staticmethod
    def detect_water_patterns(bp_seq):
        P=[]; changes=sum(1 for i in range(1,len(bp_seq)) if bp_seq[i]!=bp_seq[i-1])
        vol = changes/len(bp_seq)
        if vol < 0.3: P.append("静水路")
        elif vol < 0.6: P.append("微澜路")
        else: P.append("激流路")
        return P

    # 特殊格局
    @staticmethod
    def detect_special_patterns(bp_seq):
        P=[]; S=AdvancedPatternDetector.get_streaks(bp_seq)
        if len(S)>=3 and (S[-3]>=3 and S[-2]==1 and S[-1]>=3): P.append("回头龙")
        b_ratio = bp_seq.count('B')/len(bp_seq)
        if b_ratio>0.7: P.append("庄王格局")
        elif b_ratio<0.3: P.append("闲霸格局")
        elif 0.45<=b_ratio<=0.55: P.append("平衡格局")
        return P

# ------------------------ 看路推荐 ------------------------
def road_recommendation(roads):
    lines = []
    final = ""
    # 大路
    if roads['big_road']:
        last_col = roads['big_road'][-1]
        color_cn = "庄" if last_col[-1] == 'B' else "闲"
        streak = len(last_col)
        if streak >= 3:
            lines.append(f"大路：{color_cn}连{streak}局 → 顺路{color_cn}")
            final = f"顺大路{color_cn}"
        else:
            lines.append(f"大路：{color_cn}走势平衡")

    # 大眼路
    if roads['big_eye_road']:
        r = roads['big_eye_road'].count('R'); b = roads['big_eye_road'].count('B')
        if r > b: lines.append("大眼路：红>蓝 → 趋势延续")
        elif b > r: lines.append("大眼路：蓝>红 → 有反转迹象")
        else: lines.append("大眼路：红=蓝 → 稳定期")

    # 小路
    if roads['small_road']:
        r = roads['small_road'].count('R'); b = roads['small_road'].count('B')
        if r > b: lines.append("小路：红>蓝 → 延续趋势")
        elif b > r: lines.append("小路：蓝>红 → 节奏转弱")
        else: lines.append("小路：红=蓝 → 平衡")

    # 蟑螂路
    if roads['cockroach_road']:
        last3 = roads['cockroach_road'][-3:]
        if last3:
            trend = "红红蓝" if last3.count('R') == 2 else ("蓝蓝红" if last3.count('B') == 2 else "混乱")
            lines.append(f"蟑螂路：{trend} → {'轻微震荡' if trend!='混乱' else '趋势不明'}")

    if not final:
        if roads['big_eye_road']:
            r = roads['big_eye_road'].count('R'); b = roads['big_eye_road'].count('B')
            if r > b: final = "顺路（偏红，延续）"
            elif b > r: final = "反路（偏蓝，注意反转）"
            else: final = "暂无明显方向"
        else:
            final = "暂无明显方向"
    return {"lines": lines, "final": final}

# ------------------------ 状态信号（突破/共振/衰竭） ------------------------
class GameStateDetector:
    @staticmethod
    def detect_high_probability_moments(roads):
        signals = []
        br = GameStateDetector._detect_road_breakthrough(roads['big_road'])
        if br: signals.append(f"大路突破-{br}")
        res = GameStateDetector._detect_multi_road_alignment(roads)
        if res: signals.append(f"多路共振-{res}")
        ex = GameStateDetector._detect_streak_exhaustion(roads)
        if ex: signals.append(f"连势衰竭-{ex}")
        return signals

    @staticmethod
    def _detect_road_breakthrough(big_road):
        if len(big_road) < 4: return None
        last_4 = big_road[-4:]
        lens = [len(c) for c in last_4]
        last_color = last_4[-1][-1] if last_4[-1] else None
        if not last_color: return None
        color_cn = "庄" if last_color=='B' else "闲"
        if (lens[-1] > max(lens[-4:-1]) + 1 and all(l <= 2 for l in lens[-4:-1])):
            return f"{color_cn}势突破"
        if (lens[-4] < lens[-3] < lens[-2] < lens[-1]):
            return f"{color_cn}势加速"
        return None

    @staticmethod
    def _detect_multi_road_alignment(roads):
        sig=[]
        if roads['big_road'] and roads['big_road'][-1]:
            if len(roads['big_road'][-1])>=3: sig.append(roads['big_road'][-1][-1])
        if roads['big_eye_road']:
            last3 = roads['big_eye_road'][-3:]
            if last3 and all(x=='R' for x in last3): sig.append('B')
            elif last3 and all(x=='B' for x in last3): sig.append('P')
        if roads['small_road']:
            last3 = roads['small_road'][-3:]
            if last3 and len(set(last3))==1: sig.append('B' if last3[0]=='R' else 'P')
        if sig:
            c = Counter(sig).most_common(1)[0]
            if c[1] >= 2: return "庄趋势" if c[0]=='B' else "闲趋势"
        return None

    @staticmethod
    def _detect_streak_exhaustion(roads):
        bead = roads['bead_road']
        if not bead: return None
        cur = bead[-1]; streak=1
        for i in range(len(bead)-2,-1,-1):
            if bead[i]==cur: streak+=1
            else: break
        if streak<5: return None
        rev=0
        if len(roads['big_eye_road'])>=2 and roads['big_eye_road'][-1]!=roads['big_eye_road'][-2]: rev+=1
        if roads['small_road'] and sum(1 for x in roads['small_road'][-3:] if x!=roads['small_road'][-1])>=2: rev+=1
        if rev>=1: return f"{'庄' if cur=='B' else '闲'}龙衰竭"
        return None

# ------------------------ Hybrid 数值（含 EOR） ------------------------
def compute_hybrid_metrics():
    if not st.session_state.ultimate_games:
        return {"Hybrid":0,"Z":0,"CUSUM":0,"Bayes":0,"Mom":0,"Ratio":0,"MC":0,"EOR":0}
    seq = [g['result'] for g in st.session_state.ultimate_games if g['result'] in ['B','P']]
    if not seq:
        return {"Hybrid":0,"Z":0,"CUSUM":0,"Bayes":0,"Mom":0,"Ratio":0,"MC":0,"EOR":0}

    # Ratio / Z
    b_ratio = seq.count('B')/len(seq)
    z_score = 0 if len(seq)<10 else (b_ratio-0.5)/max(1e-6, np.sqrt(0.25/len(seq)))  # 标准误

    # CUSUM
    cusum = sum(1 if s=='B' else -1 for s in seq)/len(seq)

    # Momentum: 最近4步与最后一步一致度
    if len(seq)<4: mom = 0
    else:
        recent = seq[-4:]
        mom = (sum(1 for x in recent if x==recent[-1])/4) - 0.5

    # Bayes（信息量随局数单调增长，压缩到0~1）
    bayes = np.tanh(len(seq)/120)

    # EOR（按副数：越少表示牌靴越深，偏移风险上升）——展示向
    eor = (8 - st.session_state["eor_decks"])/8 if st.session_state["eor_decks"]>0 else 0

    # MC（示意：动量*贝叶斯置信）
    mc = mom*bayes

    # Hybrid（展示用综合分，不强行介入主决策）
    hybrid = (z_score*0.25 + cusum*0.35 + mom*0.2 + eor*0.2)

    return {"Hybrid":hybrid,"Z":z_score,"CUSUM":cusum,"Bayes":bayes,"Mom":mom,"Ratio":b_ratio,"MC":mc,"EOR":eor}

# ------------------------ 分析引擎（方向 + 置信度） ------------------------
class UltimateAnalysisEngine:
    @staticmethod
    def comprehensive_analysis(sequence):
        if len(sequence)<4:
            return {"direction":"HOLD","confidence":0.5,"reason":"数据不足，请记录更多牌局",
                    "patterns":[],"risk_level":"medium","risk_text":"🟡 中风险",
                    "current_streak":0,"volatility":0,"state_signals":[]}

        bp = [x for x in sequence if x in ['B','P']]
        patterns = AdvancedPatternDetector.detect_all_patterns(sequence)

        # 趋势
        b_ratio = bp.count('B')/len(bp)
        recent = bp[-8:] if len(bp)>=8 else bp
        b_recent = recent.count('B')/len(recent)

        # 连势
        cur = bp[-1]; streak=1
        for i in range(len(bp)-2,-1,-1):
            if bp[i]==cur: streak+=1
            else: break

        # 动能
        changes = sum(1 for i in range(1,len(bp)) if bp[i]!=bp[i-1])
        vol = changes/len(bp)
        if len(bp)>=4:
            mom = (sum(1 for x in bp[-4:] if x==bp[-1])/4) - 0.5
        else:
            mom = 0

        # 融合得分（与旧版一致）
        base = 0
        if patterns: base += len(patterns)*0.1
        base += 0.3 if b_ratio>0.6 else (-0.3 if b_ratio<0.4 else 0)
        base += 0.2 if b_recent>0.75 else (-0.2 if b_recent<0.25 else 0)
        if streak>=3: base += (streak*0.1 if cur=='B' else -streak*0.1)
        base += mom*0.2

        # 初步方向
        if base>0.15: direction="B"
        elif base<-0.15: direction="P"
        else: direction="HOLD"

        # 置信度
        conf = 0.5 if direction=="HOLD" else min(0.9, 0.5 + abs(base)*0.4 + min(0.3,0.1*len(patterns)))

        # 状态信号增强
        signals = GameStateDetector.detect_high_probability_moments(st.session_state.expert_roads)
        if signals:
            direction, conf = UltimateAnalysisEngine._apply_state_enhancement(direction, conf, signals)

        # 风险
        risk_level, risk_text = ProfessionalRiskManager.get_risk_level(conf, vol)

        # 理由
        reason = UltimateAnalysisEngine._reason(patterns, direction, streak, risk_level, signals)

        return {"direction":direction,"confidence":conf,"reason":reason,"patterns":patterns,
                "risk_level":risk_level,"risk_text":risk_text,"current_streak":streak,
                "volatility":vol,"state_signals":signals}

    @staticmethod
    def _apply_state_enhancement(direction, confidence, signals):
        d, c = direction, confidence
        for s in signals:
            if ('突破' in s) or ('共振' in s):
                c = min(0.95, c*1.25)
                if '庄' in s and d!='B': d='B'
                if '闲' in s and d!='P': d='P'
            elif '衰竭' in s and d!='HOLD':
                d='HOLD'; c=max(c,0.6)
        return d, c

    @staticmethod
    def _reason(patterns, direction, streak, risk_level, signals):
        parts=[]
        if patterns: parts.append(f"模式:{','.join(patterns[:3])}")
        if streak>=2: parts.append(f"连{streak}局")
        if signals: parts.append(f"状态:{','.join(signals[:2])}")
        parts.append(f"风险:{risk_level}")
        if direction=="HOLD": parts.append("建议观望")
        return " | ".join(parts)

# ------------------------ 风控 ------------------------
class ProfessionalRiskManager:
    @staticmethod
    def calculate_position_size(confidence, streak):
        base = 1.0
        if confidence>0.8: base*=1.2
        elif confidence>0.7: base*=1.0
        elif confidence>0.6: base*=0.8
        else: base*=0.5
        if streak>=3: base*=1.1
        # 连输保护
        if st.session_state.risk_data.get('consecutive_losses',0) >= 2: base*=0.7
        if st.session_state.risk_data.get('consecutive_losses',0) >= 3: base*=0.5
        return min(base, 2.0)

    @staticmethod
    def get_risk_level(confidence, volatility):
        score = (1-confidence) + volatility
        if score < 0.3: return "low","🟢 低风险"
        if score < 0.6: return "medium","🟡 中风险"
        if score < 0.8: return "high","🟠 高风险"
        return "extreme","🔴 极高风险"

    @staticmethod
    def get_trading_suggestion(risk_level, direction):
        s = {
            "low":{"B":"✅ 庄势明确，可适度加仓","P":"✅ 闲势明确，可适度加仓","HOLD":"⚪ 趋势平衡，正常操作"},
            "medium":{"B":"⚠️ 庄势一般，建议轻仓","P":"⚠️ 闲势一般，建议轻仓","HOLD":"⚪ 信号不明，建议观望"},
            "high":{"B":"🚨 高波动庄势，谨慎操作","P":"🚨 高波动闲势，谨慎操作","HOLD":"⛔ 高风险期，建议休息"},
            "extreme":{"B":"⛔ 极高风险，强烈建议观望","P":"⛔ 极高风险，强烈建议观望","HOLD":"⛔ 市场混乱，暂停交易"}
        }
        return s[risk_level].get(direction,"正常操作")

# ------------------------ 输入界面 ------------------------
def display_card_input():
    col1, col2 = st.columns(2)
    with col1:
        player_input = st.text_input("闲家牌", placeholder="K10 或 552", key="player_card")
    with col2:
        banker_input = st.text_input("庄家牌", placeholder="55 或 AJ", key="banker_card")

    # EOR 副数设置（无冲突版）
    st.markdown("### ⚙️ EOR 副数设置")
    st.number_input("🛠️ EOR 副数 (>0 启用)", min_value=0, max_value=8, step=1, key="eor_decks")
    st.caption(f"当前 EOR 副数: {st.session_state['eor_decks']} 副牌")

    st.markdown("### 🏆 本局结果")
    c1, c2, c3 = st.columns(3)
    with c1: banker_btn = st.button("🔴 庄赢", use_container_width=True, type="primary")
    with c2: player_btn = st.button("🔵 闲赢", use_container_width=True)
    with c3: tie_btn = st.button("⚪ 和局", use_container_width=True)

    if banker_btn or player_btn or tie_btn:
        p = parse_cards(player_input); b = parse_cards(banker_input)
        if len(p)>=2 and len(b)>=2:
            result = 'B' if banker_btn else ('P' if player_btn else 'T')
            record_game(result, p, b, 'card')
        else:
            st.error("❌ 至少输入两张牌")

def display_quick_input():
    st.info("💡 快速模式：直接记录结果，用于快速看路分析")
    c1, c2 = st.columns(2)
    with c1: quick_b = st.button("🔴 庄赢", use_container_width=True, key="qb")
    with c2: quick_p = st.button("🔵 闲赢", use_container_width=True, key="qp")
    st.markdown("### 📝 批量输入")
    batch_input = st.text_input("输入BP序列", placeholder="BPBBP 或 庄闲庄庄闲", key="batch_input")
    if st.button("✅ 确认批量输入", use_container_width=True, key="batch_ok") and batch_input:
        s = batch_input.upper().replace('庄','B').replace('闲','P').replace(' ','')
        vals = [c for c in s if c in ['B','P']]
        for r in vals: record_game(r, ['X','X'], ['X','X'], 'batch', rerun=False)
        st.success(f"✅ 批量添加{len(vals)}局"); st.rerun()
    if quick_b or quick_p:
        record_game('B' if quick_b else 'P', ['X','X'], ['X','X'], 'quick')

def record_game(result, p_cards, b_cards, mode, rerun=True):
    g = {'round': len(st.session_state.ultimate_games)+1,
         'player_cards': p_cards, 'banker_cards': b_cards,
         'result': result, 'mode': mode, 'time': datetime.now().strftime("%H:%M")}
    st.session_state.ultimate_games.append(g)
    if result in ['B','P']:
        # 路子更新
        CompleteRoadAnalyzer.update_all_roads(result)
        # 风控 streak 仅用作展示（不影响方向）
        st.session_state.risk_data['win_streak'] = st.session_state.risk_data.get('win_streak',0) + 1
        st.session_state.risk_data['consecutive_losses'] = 0
    else:
        st.session_state.risk_data['consecutive_losses'] = st.session_state.risk_data.get('consecutive_losses',0)+1
        st.session_state.risk_data['win_streak'] = 0
    st.success(f"✅ 第{g['round']}局记录成功！")
    if rerun: st.rerun()

# ------------------------ 展示：智能分析（含推荐/状态/Hybrid/风控） ------------------------
def display_complete_analysis():
    if len(st.session_state.ultimate_games) < 3:
        st.info("🎲 请先记录至少3局牌局数据"); return

    seq = [g['result'] for g in st.session_state.ultimate_games]
    # Hybrid 数值
    metrics = compute_hybrid_metrics()
    st.markdown(f"""
    <div style="background:#1a1a1a;padding:10px;border-radius:10px;margin-top:10px;color:white;text-align:center;">
    <b>📊 Hybrid 数据</b><br>
    Hybrid:{metrics['Hybrid']:+.2f} | Z:{metrics['Z']:+.2f}σ | CUSUM:{metrics['CUSUM']:+.2f} | Bayes:{metrics['Bayes']:+.2f} | Mom:{metrics['Mom']:+.2f} | Ratio:{metrics['Ratio']:.2f} | MC:{metrics['MC']:+.2f} | EOR:{metrics['EOR']:+.2f}
    </div>
    """, unsafe_allow_html=True)

    # 看路推荐
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

    # 主决策
    analysis = UltimateAnalysisEngine.comprehensive_analysis(seq)

    # 状态信号
    if analysis.get('state_signals'):
        for s in analysis['state_signals']:
            st.markdown(f"""<div class="state-signal">🚀 状态信号：{s}</div>""", unsafe_allow_html=True)

    # 预测卡片
    direction, confidence = analysis['direction'], analysis['confidence']
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

    # 模式徽章
    if patterns:
        st.markdown("### 🧩 检测模式")
        html = "".join([f'<span class="pattern-badge">{p}</span>' for p in patterns[:5]])
        st.markdown(html, unsafe_allow_html=True)

    # 风控
    display_risk_panel(analysis)

def display_risk_panel(analysis):
    st.markdown("### 🛡️ 风险控制")
    pos = ProfessionalRiskManager.calculate_position_size(analysis['confidence'], analysis.get('current_streak',0))
    sug = ProfessionalRiskManager.get_trading_suggestion(analysis['risk_level'], analysis['direction'])
    st.markdown(f"""
    <div class="risk-panel">
        <h4 style="color: white; margin: 0 0 10px 0;">📊 风险控制建议</h4>
        <p style="color: #ccc; margin: 5px 0;"><strong>仓位建议:</strong> {pos:.1f}倍基础仓位</p>
        <p style="color: #ccc; margin: 5px 0;"><strong>操作建议:</strong> {sug}</p>
        <p style="color: #ccc; margin: 5px 0;"><strong>连赢:</strong> {st.session_state.risk_data.get('win_streak',0)}局 | <strong>连输:</strong> {st.session_state.risk_data.get('consecutive_losses',0)}局</p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------ 六路显示 ------------------------
def display_complete_roads():
    R = st.session_state.expert_roads
    st.markdown("## 🛣️ 完整六路分析")

    st.markdown("#### 🟠 珠路 (最近20局)")
    if R['bead_road']:
        bead_display = " ".join(["🔴" if x=='B' else "🔵" for x in R['bead_road'][-20:]])
        st.markdown(f'<div class="road-display">{bead_display}</div>', unsafe_allow_html=True)

    st.markdown("#### 🔴 大路")
    if R['big_road']:
        for i, col in enumerate(R['big_road'][-6:]):
            col_display = " ".join(["🔴" if x=='B' else "🔵" for x in col])
            st.markdown(f'<div class="multi-road">第{i+1}列: {col_display}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if R['big_eye_road']:
            st.markdown("#### 👁️ 大眼路")
            display = " ".join(["🔴" if x=='R' else "🔵" for x in R['big_eye_road'][-12:]])
            st.markdown(f'<div class="multi-road">{display}</div>', unsafe_allow_html=True)
    with col2:
        if R['small_road']:
            st.markdown("#### 🔵 小路")
            display = " ".join(["🔴" if x=='R' else "🔵" for x in R['small_road'][-10:]])
            st.markdown(f'<div class="multi-road">{display}</div>', unsafe_allow_html=True)

    if R['three_bead_road']:
        st.markdown("#### 🔶 三珠路")
        for i, group in enumerate(R['three_bead_road'][-6:]):
            display = " ".join(["🔴" if x=='B' else "🔵" for x in group])
            st.markdown(f'<div class="multi-road">第{i+1}组: {display}</div>', unsafe_allow_html=True)

# ------------------------ 统计 ------------------------
def display_professional_stats():
    if not st.session_state.ultimate_games:
        st.info("暂无统计数据"); return
    games = st.session_state.ultimate_games
    results = [g['result'] for g in games]
    bead = st.session_state.expert_roads['bead_road']

    st.markdown("## 📊 专业统计")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("总局数", len(results))
    with c2: st.metric("庄赢", results.count('B'))
    with c3: st.metric("闲赢", results.count('P'))
    with c4: st.metric("和局", results.count('T'))

    if bead:
        st.markdown("#### 📈 高级分析")
        c1,c2,c3 = st.columns(3)
        with c1:
            total = len([r for r in results if r in ['B','P']])
            st.metric("庄胜率", f"{(results.count('B')/total*100):.1f}%") if total>0 else st.metric("庄胜率","-")
        with c2:
            avg_streak = np.mean([len(list(g)) for k,g in groupby(bead)]) if len(bead)>0 else 0
            st.metric("平均连赢", f"{avg_streak:.1f}局")
        with c3:
            if len(bead)>1:
                changes = sum(1 for i in range(1,len(bead)) if bead[i]!=bead[i-1])
                vol = changes/len(bead)*100
                st.metric("波动率", f"{vol:.1f}%")

# ------------------------ 历史 ------------------------
def display_complete_history():
    if not st.session_state.ultimate_games:
        st.info("暂无历史记录"); return
    st.markdown("## 📝 完整历史（最近10局）")
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
    st.markdown('<h1 class="main-header">🐉 Baccarat Master Precision 12 — Hybrid + EOR Control (Full Route Vision)</h1>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 智能分析", "🛣️ 六路分析", "📊 专业统计", "📝 历史记录"])

    with tab1:
        # 输入区：牌点 & 快速看路
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            if st.button("🃏 牌点输入", use_container_width=True, type="primary"):
                st.session_state.input_mode = "card"; st.rerun()
        with mode_col2:
            if st.button("🎯 快速看路", use_container_width=True):
                st.session_state.input_mode = "result"; st.rerun()
        if "input_mode" not in st.session_state:
            st.session_state.input_mode = "card"
        if st.session_state.input_mode == "card":
            display_card_input()
        else:
            display_quick_input()

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
            st.session_state.expert_roads = {k: ([] if k!='three_bead_road' else []) for k in st.session_state.expert_roads}
            st.session_state.risk_data = {'consecutive_losses': 0, 'win_streak': 0}
            st.success("新牌靴开始！"); st.rerun()
    with c2:
        if st.button("📋 导出数据（占位）", use_container_width=True):
            st.info("数据导出功能准备中...")

if __name__ == "__main__":
    main()
