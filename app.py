# -*- coding: utf-8 -*-
# Precision 12 Hybrid + EOR Control Ultimate
# 保留全部功能：看路推荐 / 状态检测 / 六路 / 风控 / 牌点增强
# 仅新增：
# 1) EOR 副数可调（牌点模式启用时生效）
# 2) Hybrid 数值显示行（Hybrid / Z / CUSUM / Bayes / Mom / Ratio / MC / EOR）
#   —— 显示用，不改变原有方向与置信度计算逻辑

import streamlit as st
import numpy as np
from collections import defaultdict, Counter
from datetime import datetime
from itertools import groupby

st.set_page_config(page_title="百家乐大师终极版", layout="centered")

# ---------------------------- 样式 ----------------------------
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
    .enhancement-panel {
        background: #2d3748;
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #00D4AA;
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
    .hybrid-line {
        font-family: monospace;
        color: #fff;
        margin-top: 8px;
        opacity: .95;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐉 Precision 12 Hybrid + EOR Control Ultimate</h1>', unsafe_allow_html=True)

# ---------------------------- 状态 ----------------------------
if "ultimate_games" not in st.session_state:
    st.session_state.ultimate_games = []
if "expert_roads" not in st.session_state:
    st.session_state.expert_roads = {
        'big_road': [], 'bead_road': [], 'big_eye_road': [],
        'small_road': [], 'cockroach_road': [], 'three_bead_road': []
    }
if "risk_data" not in st.session_state:
    st.session_state.risk_data = {'current_level':'medium','position_size':1.0,'stop_loss':3,'consecutive_losses':0,'win_streak':0}
# CUSUM 累计（仅用于显示，不参与原决策逻辑）
if "cusum_meter" not in st.session_state:
    st.session_state.cusum_meter = 0.0
# EOR 副数（>0 启用），默认 8（可调）
if "eor_decks" not in st.session_state:
    st.session_state.eor_decks = 8

# ---------------------------- 六路分析 ----------------------------
class CompleteRoadAnalyzer:
    @staticmethod
    def update_all_roads(result):
        if result not in ['B','P']: return
        roads = st.session_state.expert_roads
        roads['bead_road'].append(result)
        if not roads['big_road']:
            roads['big_road'].append([result])
        else:
            last_col = roads['big_road'][-1]
            if last_col[-1] == result: last_col.append(result)
            else: roads['big_road'].append([result])

        if len(roads['big_road']) >= 2:
            big_eye = []
            for i in range(1, len(roads['big_road'])):
                big_eye.append('R' if len(roads['big_road'][i]) >= len(roads['big_road'][i-1]) else 'B')
            roads['big_eye_road'] = big_eye[-20:]

        if len(roads['big_eye_road']) >= 2:
            small = []
            for i in range(1, len(roads['big_eye_road'])):
                small.append('R' if roads['big_eye_road'][i] == roads['big_eye_road'][i-1] else 'B')
            roads['small_road'] = small[-15:]

        if len(roads['small_road']) >= 2:
            cock = []
            for i in range(1, len(roads['small_road'])):
                cock.append('R' if roads['small_road'][i] == roads['small_road'][i-1] else 'B')
            roads['cockroach_road'] = cock[-12:]

        bead = roads['bead_road']
        if len(bead) >= 3:
            groups = [bead[i:i+3] for i in range(0, len(bead)-2, 3)]
            roads['three_bead_road'] = groups[-8:]

# ---------------------------- 模式检测（节选，含 60+ 分类核心） ----------------------------
class AdvancedPatternDetector:
    @staticmethod
    def detect_all_patterns(sequence):
        bp = [x for x in sequence if x in ['B','P']]
        if len(bp) < 4: return []
        p = []
        try:
            p += AdvancedPatternDetector.detect_dragon_patterns(bp)
            p += AdvancedPatternDetector.detect_jump_patterns(bp)
            p += AdvancedPatternDetector.detect_house_patterns(bp)
            p += AdvancedPatternDetector.detect_trend_patterns(bp)
            p += AdvancedPatternDetector.detect_special_patterns(bp)
            p += AdvancedPatternDetector.detect_water_patterns(bp)
        except Exception:
            if len(set(bp[-4:])) == 1: p.append(f"{bp[-1]}长龙")
        return p[:8]

    @staticmethod
    def get_streaks(bp):
        if not bp: return []
        s, cur, c = [], bp[0], 1
        for i in range(1,len(bp)):
            if bp[i]==cur: c+=1
            else: s.append(c); cur=bp[i]; c=1
        s.append(c); return s

    @staticmethod
    def detect_dragon_patterns(bp):
        p=[]
        if len(set(bp[-4:]))==1: p.append(f"{bp[-1]}长龙")
        if len(bp)>=5 and len(set(bp[-5:]))==1: p.append(f"强{bp[-1]}长龙")
        if len(bp)>=6 and len(set(bp[-6:]))==1: p.append(f"超强{bp[-1]}长龙")
        return p

    @staticmethod
    def detect_jump_patterns(bp):
        p=[]
        if len(bp)>=6 and bp[-6:] in [['B','P','B','P','B','P'],['P','B','P','B','P','B']]: p.append("完美单跳")
        if len(bp)>=8 and bp[-8:] in [['B','B','P','P','B','B','P','P'],['P','P','B','B','P','P','B','B']]: p.append("齐头双跳")
        if len(bp)>=5 and bp[-5:] in [['B','P','B','P','B'],['P','B','P','B','P']]: p.append("长短单跳")
        return p

    @staticmethod
    def detect_house_patterns(bp):
        p=[]; s=AdvancedPatternDetector.get_streaks(bp)
        if len(s)>=3 and (s[-3]==2 and s[-2]==1 and s[-1]==2): p.append("一房一厅")
        if len(s)>=4 and (s[-4]==2 and s[-3]==2 and s[-2]==1 and s[-1]==2): p.append("两房一厅")
        if len(s)>=4 and (s[-4]>=3 and s[-3]>=3 and s[-2]==1 and s[-1]>=3): p.append("三房一厅")
        if len(s)>=4 and (s[-4]>=4 and s[-3]>=4 and s[-2]==1 and s[-1]>=4): p.append("四房一厅")
        if len(s)>=4 and (s[-4]>=3 and s[-3]>=3 and s[-2]==1 and s[-1]==2): p.append("假三房")
        return p

    @staticmethod
    def detect_trend_patterns(bp):
        p=[]; s=AdvancedPatternDetector.get_streaks(bp)
        if len(s)>=4 and all(s[i]<s[i+1] for i in range(-4,-1)): p.append("上山路")
        if len(s)>=4 and all(s[i]>s[i+1] for i in range(-4,-1)): p.append("下山路")
        if len(s)>=5 and ((s[-5]<s[-4]>s[-3]<s[-2]>s[-1]) or (s[-5]>s[-4]<s[-3]>s[-2]<s[-1])): p.append("楼梯路")
        return p

    @staticmethod
    def detect_water_patterns(bp):
        p=[]; ch=sum(1 for i in range(1,len(bp)) if bp[i]!=bp[i-1]); vol=ch/len(bp)
        if vol<0.3: p.append("静水路")
        elif vol<0.6: p.append("微澜路")
        else: p.append("激流路")
        return p

    @staticmethod
    def detect_special_patterns(bp):
        p=[]; s=AdvancedPatternDetector.get_streaks(bp)
        if len(s)>=3 and (s[-3]>=3 and s[-2]==1 and s[-1]>=3): p.append("回头龙")
        b_ratio = bp.count('B')/len(bp)
        if b_ratio>0.7: p.append("庄王格局")
        elif b_ratio<0.3: p.append("闲霸格局")
        elif 0.45<=b_ratio<=0.55: p.append("平衡格局")
        return p

# ---------------------------- 牌点增强（原样保留） ----------------------------
class CardEnhancementAnalyzer:
    @staticmethod
    def analyze_card_enhancement(games):
        card_games=[g for g in games if g.get('mode')=='card' and len(g['player_cards'])>=2 and len(g['banker_cards'])>=2]
        if len(card_games)<2: return {"enhancement_factor":0, "reason":"牌点数据不足"}
        enh=0; reasons=[]
        # 天牌
        recent=card_games[-3:]; cnt=0
        for g in recent:
            p=CardEnhancementAnalyzer._pts(g['player_cards']); b=CardEnhancementAnalyzer._pts(g['banker_cards'])
            if p>=8 or b>=8: cnt+=1
        if cnt>=2: enh+=0.08; reasons.append(f"天牌密集({cnt}局)")
        elif cnt==1: enh+=0.03; reasons.append("天牌出现")
        # 点数动量
        if len(card_games)>=4:
            pts=[]; 
            for g in card_games[-4:]:
                pts += [CardEnhancementAnalyzer._pts(g['player_cards']), CardEnhancementAnalyzer._pts(g['banker_cards'])]
            avg=sum(pts)/len(pts)
            if avg<4: enh+=0.06; reasons.append("小点数期")
            elif avg>7: enh-=0.04; reasons.append("大点数期")
        # 补牌密度
        total=min(10,len(card_games)); draw=sum(1 for g in card_games[-total:] if (CardEnhancementAnalyzer._pts(g['player_cards'])<6 or CardEnhancementAnalyzer._pts(g['banker_cards'])<6))
        ratio=draw/total
        if ratio>0.7: enh-=0.05; reasons.append("补牌密集")
        elif ratio<0.3: enh+=0.04; reasons.append("补牌稀少")
        return {"enhancement_factor":max(-0.2,min(0.2,enh)), "reason":" | ".join(reasons) if reasons else "无增强信号"}

    @staticmethod
    def _pts(cards):
        mp={'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':0,'J':0,'Q':0,'K':0}
        return sum(mp.get(c,0) for c in cards)%10

# ---------------------------- EOR 计算（显示用，不改决策） ----------------------------
class EORCalculator:
    """根据已记录牌点与设定副数，估算剩余 0 点牌(10/J/Q/K) 与 1-9 点牌的比例偏差。"""
    RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
    ZERO = set(['10','J','Q','K'])
    @staticmethod
    def shoe_counts(decks:int):
        per_rank = 4*decks  # 每种点数每副有4张*副数
        return {r: per_rank for r in EORCalculator.RANKS}

    @staticmethod
    def from_games(games, decks:int):
        if decks<=0: 
            return {"bias":0.0, "depth":0.0}
        counts = EORCalculator.shoe_counts(decks)
        seen=0
        for g in games:
            if g.get('mode')=='card':
                for c in g['player_cards']+g['banker_cards']:
                    if c in counts and counts[c]>0:
                        counts[c]-=1; seen+=1
        remain = sum(counts.values())
        if remain<=0: return {"bias":0.0, "depth":1.0}
        zero_left = sum(counts[r] for r in EORCalculator.ZERO)
        nonzero_left = remain - zero_left
        bias = (zero_left - nonzero_left)/remain  # 正：0点牌多；负：1-9多
        depth = seen / (seen + remain)
        return {"bias": float(bias), "depth": float(depth)}

# ---------------------------- 状态检测器（保持原样） ----------------------------
class GameStateDetector:
    @staticmethod
    def detect_high_probability_moments(roads):
        hp=[]
        br=GameStateDetector._breakthrough(roads['big_road'])
        if br: hp.append(f"大路突破-{br}")
        rs=GameStateDetector._resonance(roads); 
        if rs: hp.append(f"多路共振-{rs}")
        ex=GameStateDetector._exhaust(roads); 
        if ex: hp.append(f"连势衰竭-{ex}")
        return hp

    @staticmethod
    def _breakthrough(big):
        if len(big)<4: return None
        last4=big[-4:]; lens=[len(c) for c in last4]
        col=last4[-1][-1]
        cn="庄" if col=='B' else "闲"
        if (lens[-1] > max(lens[-4:-1])+1 and all(l<=2 for l in lens[-4:-1])): return f"{cn}势突破"
        if (lens[-4] < lens[-3] < lens[-2] < lens[-1]): return f"{cn}势加速"
        return None

    @staticmethod
    def _resonance(roads):
        sig=[]
        if roads['big_road'] and roads['big_road'][-1] and len(roads['big_road'][-1])>=3:
            sig.append(roads['big_road'][-1][-1])
        if roads['big_eye_road']:
            eye=roads['big_eye_road'][-3:]
            if eye and all(x=='R' for x in eye): sig.append('B')
            elif eye and all(x=='B' for x in eye): sig.append('P')
        if roads['small_road']:
            sm=roads['small_road'][-3:]
            if sm and len(set(sm))==1: sig.append('B' if sm[0]=='R' else 'P')
        if sig:
            c=Counter(sig).most_common(1)[0]
            if c[1]>=2: return "庄趋势" if c[0]=='B' else "闲趋势"
        return None

    @staticmethod
    def _exhaust(roads):
        bead=roads['bead_road']
        if not roads['big_road'] or not bead: return None
        cur = bead[-1]; streak=1
        for i in range(len(bead)-2,-1,-1):
            if bead[i]==cur: streak+=1
            else: break
        if streak<5: return None
        cn="庄" if cur=='B' else "闲"
        rev=0
        if len(roads['big_eye_road'])>=2 and roads['big_eye_road'][-1]!=roads['big_eye_road'][-2]: rev+=1
        if roads['small_road'] and sum(1 for x in roads['small_road'][-3:] if x!=roads['small_road'][-1])>=2: rev+=1
        if rev>=1: return f"{cn}龙衰竭"
        return None

# ---------------------------- 风控 ----------------------------
class ProfessionalRiskManager:
    @staticmethod
    def calculate_position_size(confidence, streak_info):
        base=1.0
        if confidence>0.8: base*=1.2
        elif confidence>0.7: base*=1.0
        elif confidence>0.6: base*=0.8
        else: base*=0.5
        if streak_info.get('current_streak',0)>=3: base*=1.1
        if st.session_state.risk_data['consecutive_losses']>=3: base*=0.5
        elif st.session_state.risk_data['consecutive_losses']>=2: base*=0.7
        return min(base,2.0)

    @staticmethod
    def get_risk_level(confidence, volatility):
        score=(1-confidence)+volatility
        if score<0.3: return "low","🟢 低风险"
        if score<0.6: return "medium","🟡 中风险"
        if score<0.8: return "high","🟠 高风险"
        return "extreme","🔴 极高风险"

    @staticmethod
    def get_trading_suggestion(level, direction):
        s={
            "low":{"B":"✅ 庄势明确，可适度加仓","P":"✅ 闲势明确，可适度加仓","HOLD":"⚪ 趋势平衡，正常操作"},
            "medium":{"B":"⚠️ 庄势一般，建议轻仓","P":"⚠️ 闲势一般，建议轻仓","HOLD":"⚪ 信号不明，建议观望"},
            "high":{"B":"🚨 高波动庄势，谨慎操作","P":"🚨 高波动闲势，谨慎操作","HOLD":"⛔ 高风险期，建议休息"},
            "extreme":{"B":"⛔ 极高风险，强烈建议观望","P":"⛔ 极高风险，强烈建议观望","HOLD":"⛔ 市场混乱，暂停交易"}
        }
        return s[level].get(direction,"正常操作")

# ---------------------------- 核心分析引擎（原逻辑保持） ----------------------------
class UltimateAnalysisEngine:
    @staticmethod
    def comprehensive_analysis(sequence):
        if len(sequence)<4:
            return {"direction":"HOLD","confidence":0.5,"reason":"数据不足，请记录更多牌局","patterns":[],"risk_level":"medium","risk_text":"🟡 中风险","current_streak":0,"volatility":0,"state_signals":[]}

        bp=[x for x in sequence if x in ['B','P']]
        patterns = AdvancedPatternDetector.detect_all_patterns(sequence)
        current_streak = UltimateAnalysisEngine.get_current_streak(bp)

        b_ratio = bp.count('B')/len(bp)
        recent = bp[-8:] if len(bp)>=8 else bp
        b_recent = recent.count('B')/len(recent) if recent else 0.5

        volatility = UltimateAnalysisEngine.calculate_volatility(bp)
        momentum = UltimateAnalysisEngine.calculate_momentum(bp)

        base=0
        if patterns: base += len(patterns)*0.1
        if b_ratio>0.6: base += 0.3
        elif b_ratio<0.4: base -= 0.3
        if b_recent>0.75: base += 0.2
        elif b_recent<0.25: base -= 0.2
        if current_streak>=3:
            d=bp[-1]; base += current_streak*0.1 if d=="B" else -current_streak*0.1
        base += momentum*0.2

        confidence = min(0.9, 0.5 + abs(base)*0.4 + (len(patterns)*0.1))
        if base>0.15: direction="B"
        elif base<-0.15: direction="P"
        else: direction, confidence = "HOLD", 0.5

        # 状态信号增强（方向/置信度微调）
        state_signals = GameStateDetector.detect_high_probability_moments(st.session_state.expert_roads)
        if state_signals:
            direction, confidence = UltimateAnalysisEngine._apply_state_enhancement(direction, confidence, state_signals, bp)

        risk_level, risk_text = ProfessionalRiskManager.get_risk_level(confidence, volatility)
        reason = UltimateAnalysisEngine.generate_reasoning(patterns, direction, current_streak, risk_level, state_signals)

        return {"direction":direction,"confidence":confidence,"reason":reason,"patterns":patterns,"risk_level":risk_level,"risk_text":risk_text,"current_streak":current_streak,"volatility":volatility,"state_signals":state_signals}

    @staticmethod
    def _apply_state_enhancement(direction, confidence, signals, bp):
        d=direction; c=confidence
        for s in signals:
            if '突破' in s or '共振' in s:
                c=min(0.95, c*1.3)
                if '庄' in s and d!='B': d='B'
                elif '闲' in s and d!='P': d='P'
            elif '衰竭' in s and d!='HOLD':
                d='HOLD'; c=0.6
        return d,c

    @staticmethod
    def get_current_streak(bp):
        if not bp: return 0
        cur=bp[-1]; st=1
        for i in range(len(bp)-2,-1,-1):
            if bp[i]==cur: st+=1
            else: break
        return st

    @staticmethod
    def calculate_volatility(bp):
        if len(bp)<2: return 0.0
        ch=sum(1 for i in range(1,len(bp)) if bp[i]!=bp[i-1]); return ch/len(bp)

    @staticmethod
    def calculate_momentum(bp):
        if len(bp)<4: return 0.0
        r=bp[-4:]; return sum(1 for x in r if x==r[-1])/len(r)-0.5

    @staticmethod
    def generate_reasoning(patterns, direction, streak, risk_level, signals):
        parts=[]
        if patterns: parts.append(f"模式:{','.join(patterns[:3])}")
        if streak>=2: parts.append(f"连{streak}局")
        if signals: parts.append(f"状态:{','.join(signals[:2])}")
        parts.append(f"风险:{risk_level}")
        if direction=="HOLD": parts.append("建议观望")
        return " | ".join(parts)

# ---------------------------- 看路推荐（原样） ----------------------------
def road_recommendation(roads):
    lines=[]; final=""
    if roads['big_road']:
        last_col=roads['big_road'][-1]; color="庄" if last_col[-1]=='B' else "闲"; st=len(last_col)
        if st>=3: lines.append(f"大路：{color}连{st}局 → 顺路{color}"); final=f"顺大路{color}"
        else: lines.append(f"大路：{color}走势平衡")
    if roads['big_eye_road']:
        r=roads['big_eye_road'].count('R'); b=roads['big_eye_road'].count('B')
        if r>b: lines.append("大眼路：红>蓝 → 趋势延续")
        elif b>r: lines.append("大眼路：蓝>红 → 有反转迹象")
        else: lines.append("大眼路：红=蓝 → 稳定期")
    if roads['small_road']:
        r=roads['small_road'].count('R'); b=roads['small_road'].count('B')
        if r>b: lines.append("小路：红>蓝 → 延续趋势")
        elif b>r: lines.append("小路：蓝>红 → 节奏转弱")
        else: lines.append("小路：红=蓝 → 平衡")
    if not final:
        if roads['big_eye_road']:
            r=roads['big_eye_road'].count('R'); b=roads['big_eye_road'].count('B')
            if r>b: final="顺路（偏红，延续）"
            elif b>r: final="反路（偏蓝，注意反转）"
            else: final="暂无明显方向"
        else: final="暂无明显方向"
    return {"lines":lines,"final":final}

# ---------------------------- Hybrid 显示指标（新增显示层） ----------------------------
def compute_hybrid_metrics(sequence, analysis):
    """返回 dict：Hybrid / Z / CUSUM / Bayes / Mom / Ratio / MC / EOR（全部仅用于显示）"""
    bp=[x for x in sequence if x in ['B','P']]
    if not bp: 
        return dict(Hybrid=0,Z=0,CUSUM=0,Bayes=0,Mom=0,Ratio=0,MC=0,EOR=0)
    # Hybrid: 用原 base 概念近似（方向强度），这里用 (confidence-0.5)*1.08 做线性化显示
    hybrid = (analysis['confidence']-0.5)*1.08
    # Z-score：最近窗口的二项偏差标准化
    win = bp[-8:] if len(bp)>=8 else bp
    n=len(win); k=win.count('B'); phat=k/n if n>0 else 0.5
    denom = np.sqrt(max(1e-9, 0.25/n)) if n>0 else 1.0
    z = (phat-0.5)/denom if n>0 else 0.0
    # Momentum（同原）
    mom = UltimateAnalysisEngine.calculate_momentum(bp)
    # Ratio：整体 B 占比相对 0.5 的偏移
    ratio = (bp.count('B')/len(bp)) - 0.5
    # CUSUM：根据相邻是否延续进行累计（显示用）
    step = 1.0 if (len(bp)>=2 and bp[-1]==bp[-2]) else -0.5
    st.session_state.cusum_meter = float(np.clip(st.session_state.cusum_meter + step, -10, 10))
    cusum = st.session_state.cusum_meter/10.0  # 归一到 -1~1
    # Bayes：用最近窗的对数几率近似显示
    eps=1e-6
    logodds = np.log((phat+eps)/(1-phat+eps))
    bayes = float(np.tanh(logodds/2))  # 压缩到 -1~1
    # 一阶 Markov（MC）显示用：转移矩阵估计同边概率-对边概率
    same=0; total=0
    for i in range(1,len(bp)):
        total+=1; 
        if bp[i]==bp[i-1]: same+=1
    mc = (same/total - 0.5) if total>0 else 0.0
    # EOR（基于副数与已见牌点）
    eor_info = EORCalculator.from_games(st.session_state.ultimate_games, st.session_state.eor_decks)
    eor = eor_info['bias']  # -1~+1（0点牌偏多为正）
    return dict(Hybrid=hybrid, Z=z, CUSUM=cusum, Bayes=bayes, Mom=mom, Ratio=ratio, MC=mc, EOR=eor)

# ---------------------------- 输入界面（加回 EOR 控件） ----------------------------
def display_complete_interface():
    st.markdown("## 🎮 双模式输入系统")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🃏 牌点输入", use_container_width=True, type="primary"):
            st.session_state.input_mode="card"; st.rerun()
    with c2:
        if st.button("🎯 快速看路", use_container_width=True):
            st.session_state.input_mode="result"; st.rerun()
    if "input_mode" not in st.session_state: st.session_state.input_mode="card"
    if st.session_state.input_mode=="card":
        display_card_input()
    else:
        display_quick_input()

def parse_cards(input_str):
    if not input_str: return []
    s=input_str.upper().replace(' ',''); out=[]; i=0
    while i<len(s):
        if i+1<len(s) and s[i:i+2]=='10': out.append('10'); i+=2
        elif s[i] in '123456789': out.append(s[i]); i+=1
        elif s[i] in ['A','J','Q','K','0']:
            mp={'A':'A','J':'J','Q':'Q','K':'K','0':'10'}
            out.append(mp[s[i]]); i+=1
        else: i+=1
    return out

def display_card_input():
    col1, col2 = st.columns(2)
    with col1:
        player_input = st.text_input("闲家牌 (K10 或 552)", key="player_card")
    with col2:
        banker_input = st.text_input("庄家牌 (55 或 AJ)", key="banker_card")

    # 🔧 EOR 副数调节（>0 启用）
    st.number_input("🛠️ EOR 副数（>0 启用）", min_value=0, max_value=12, value=st.session_state.eor_decks,
                    step=1, key="eor_decks", help="设置鞋内副数。>0 启用 EOR 估计；通常 6~8。")

    st.markdown("### 🏆 本局结果")
    c1, c2, c3 = st.columns(3)
    with c1: banker_btn = st.button("🔴 庄赢", use_container_width=True, type="primary")
    with c2: player_btn = st.button("🔵 闲赢", use_container_width=True)
    with c3: tie_btn = st.button("⚪ 和局", use_container_width=True)

    if banker_btn or player_btn or tie_btn:
        p_cards = parse_cards(player_input); b_cards = parse_cards(banker_input)
        if len(p_cards)>=2 and len(b_cards)>=2:
            result = 'B' if banker_btn else ('P' if player_btn else 'T')
            record_game(result, p_cards, b_cards, 'card')
        else:
            st.error("❌ 需要至少2张牌（例：K10 / 552）")

def display_quick_input():
    st.info("💡 快速模式：直接记录结果，用于快速看路分析")
    c1, c2 = st.columns(2)
    with c1: quick_b = st.button("🔴 庄赢", use_container_width=True, type="primary")
    with c2: quick_p = st.button("🔵 闲赢", use_container_width=True)
    st.markdown("### 📝 批量输入")
    batch = st.text_input("输入BP序列 (BPBBP 或 庄闲庄庄闲)", key="batch_input")
    if st.button("✅ 确认批量输入", use_container_width=True) and batch:
        s=batch.upper().replace('庄','B').replace('闲','P').replace(' ',''); valid=[c for c in s if c in ['B','P']]
        for r in valid: record_game(r, ['X','X'], ['X','X'], 'batch')
        st.success(f"✅ 批量添加 {len(valid)} 局")
    if quick_b or quick_p:
        record_game('B' if quick_b else 'P', ['X','X'], ['X','X'], 'quick')

def record_game(result, p_cards, b_cards, mode):
    game = {'round':len(st.session_state.ultimate_games)+1,'player_cards':p_cards,'banker_cards':b_cards,
            'result':result,'time':datetime.now().strftime("%H:%M"),'mode':mode}
    st.session_state.ultimate_games.append(game)
    if result in ['B','P']:
        CompleteRoadAnalyzer.update_all_roads(result)
    update_risk_data(result)
    st.success(f"✅ 记录成功! 第{game['round']}局")
    st.rerun()

def update_risk_data(result):
    risk=st.session_state.risk_data
    if result in ['B','P']:
        risk['win_streak']+=1; risk['consecutive_losses']=0
    else:
        risk['consecutive_losses']+=1; risk['win_streak']=0

# ---------------------------- 智能分析展示（含 Hybrid 行） ----------------------------
def display_complete_analysis():
    if len(st.session_state.ultimate_games)<3:
        st.info("🎲 请先记录至少3局牌局数据"); return
    sequence=[g['result'] for g in st.session_state.ultimate_games]
    analysis = UltimateAnalysisEngine.comprehensive_analysis(sequence)

    # 看路推荐条（原样）
    road_sug = road_recommendation(st.session_state.expert_roads)
    if road_sug.get("final"):
        st.markdown(f"""
        <div style="background: linear-gradient(90deg,#FFD70033,#FF634733); padding:10px 14px; border-radius:10px; margin-top:6px; margin-bottom:10px; border-left:5px solid #FFD700; color:#fff; font-weight:600; text-shadow:1px 1px 2px #000;">
            🛣️ 看路推荐：{road_sug['final']}
        </div>
        """, unsafe_allow_html=True)

    # 状态信号
    if analysis.get('state_signals'):
        for s in analysis['state_signals']:
            st.markdown(f'<div class="state-signal">🚀 状态信号：{s}</div>', unsafe_allow_html=True)

    # 预测卡片
    direction=analysis['direction']; conf=analysis['confidence']; reason=analysis['reason']
    risk_text=analysis.get('risk_text','🟡 中风险'); patterns=analysis.get('patterns',[])
    if direction=="B": color="#FF6B6B"; icon="🔴"; text="庄(B)"; bg="linear-gradient(135deg,#FF6B6B 0%,#C44569 100%)"
    elif direction=="P": color="#4ECDC4"; icon="🔵"; text="闲(P)"; bg="linear-gradient(135deg,#4ECDC4 0%,#44A08D 100%)"
    else: color="#FFE66D"; icon="⚪"; text="观望"; bg="linear-gradient(135deg,#FFE66D 0%,#F9A826 100%)"

    st.markdown(f"""
    <div class="prediction-card" style="background:{bg};">
        <h2 style="color:{color}; margin:0;">{icon} 大师推荐: {text}</h2>
        <h3 style="color:#fff; margin:10px 0;">🎯 置信度: {conf*100:.1f}% | {risk_text}</h3>
        <p style="color:#f8f9fa; margin:0;">{reason}</p>
    </div>
    """, unsafe_allow_html=True)

    # 🔢 新增：Hybrid 数值显示行（仅展示，不改逻辑）
    metrics = compute_hybrid_metrics(sequence, analysis)
    st.markdown(
        f"""<div class="hybrid-line">
        Hybrid:{metrics['Hybrid']:+.2f} |
        Z:{metrics['Z']:+.2f}σ |
        CUSUM:{metrics['CUSUM']:+.2f} |
        Bayes:{metrics['Bayes']:+.2f} |
        Mom:{metrics['Mom']:+.2f} |
        Ratio:{metrics['Ratio']:+.2f} |
        MC:{metrics['MC']:+.2f} |
        EOR:{metrics['EOR']:+.2f}
        </div>""", unsafe_allow_html=True)

    if patterns:
        st.markdown("### 🧩 检测模式")
        st.markdown("".join([f'<span class="pattern-badge">{p}</span>' for p in patterns[:5]]), unsafe_allow_html=True)

    display_risk_panel(analysis)

def display_risk_panel(analysis):
    st.markdown("### 🛡️ 风险控制")
    pos = ProfessionalRiskManager.calculate_position_size(analysis['confidence'], {'current_streak':analysis.get('current_streak',0)})
    sug = ProfessionalRiskManager.get_trading_suggestion(analysis['risk_level'], analysis['direction'])
    st.markdown(f"""
    <div class="risk-panel">
        <h4 style="color:white; margin:0 0 10px 0;">📊 风险控制建议</h4>
        <p style="color:#ccc; margin:5px 0;"><strong>仓位建议:</strong> {pos:.1f} 倍基础仓位</p>
        <p style="color:#ccc; margin:5px 0;"><strong>操作建议:</strong> {sug}</p>
        <p style="color:#ccc; margin:5px 0;"><strong>连赢:</strong> {st.session_state.risk_data['win_streak']} 局 | <strong>连输:</strong> {st.session_state.risk_data['consecutive_losses']} 局</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------- 六路/统计/历史（原样） ----------------------------
def display_complete_roads():
    roads=st.session_state.expert_roads
    st.markdown("## 🛣️ 完整六路分析")
    st.markdown("#### 🟠 珠路 (最近20局)")
    if roads['bead_road']:
        bead=" ".join(["🔴" if x=='B' else "🔵" for x in roads['bead_road'][-20:]])
        st.markdown(f'<div class="road-display">{bead}</div>', unsafe_allow_html=True)
    st.markdown("#### 🔴 大路")
    if roads['big_road']:
        for i,col in enumerate(roads['big_road'][-6:]):
            disp=" ".join(["🔴" if x=='B' else "🔵" for x in col])
            st.markdown(f'<div class="multi-road">第{i+1}列: {disp}</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        if roads['big_eye_road']:
            disp=" ".join(["🔴" if x=='R' else "🔵" for x in roads['big_eye_road'][-12:]])
            st.markdown("#### 👁️ 大眼路"); st.markdown(f'<div class="multi-road">{disp}</div>', unsafe_allow_html=True)
    with c2:
        if roads['small_road']:
            disp=" ".join(["🔴" if x=='R' else "🔵" for x in roads['small_road'][-10:]])
            st.markdown("#### 🔵 小路"); st.markdown(f'<div class="multi-road">{disp}</div>', unsafe_allow_html=True)
    if roads['three_bead_road']:
        st.markdown("#### 🔶 三珠路")
        for i,g in enumerate(roads['three_bead_road'][-6:]):
            disp=" ".join(["🔴" if x=='B' else "🔵" for x in g])
            st.markdown(f'<div class="multi-road">第{i+1}组: {disp}</div>', unsafe_allow_html=True)

def display_professional_stats():
    if not st.session_state.ultimate_games:
        st.info("暂无统计数据"); return
    games=st.session_state.ultimate_games; results=[g['result'] for g in games]; bead=st.session_state.expert_roads['bead_road']
    st.markdown("## 📊 专业统计")
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("总局数", len(results))
    with c2: st.metric("庄赢", results.count('B'))
    with c3: st.metric("闲赢", results.count('P'))
    with c4: st.metric("和局", results.count('T'))
    if bead:
        st.markdown("#### 📈 高级分析")
        d1,d2,d3=st.columns(3)
        with d1:
            total=len(results)
            st.metric("庄胜率", f"{results.count('B')/total*100:.1f}%")
        with d2:
            avg=np.mean([len(list(g)) for k,g in groupby(bead)]) if len(bead)>0 else 0
            st.metric("平均连赢", f"{avg:.1f}局")
        with d3:
            if len(bead)>1:
                changes=sum(1 for i in range(1,len(bead)) if bead[i]!=bead[i-1]); vol=changes/len(bead)*100
                st.metric("波动率", f"{vol:.1f}%")

def display_complete_history():
    if not st.session_state.ultimate_games:
        st.info("暂无历史记录"); return
    st.markdown("## 📝 完整历史")
    recent=st.session_state.ultimate_games[-10:]
    for g in reversed(recent):
        icon="🃏" if g.get('mode')=='card' else ("🎯" if g.get('mode')=='quick' else "📝")
        c1,c2,c3,c4,c5=st.columns([1,1,2,2,1])
        with c1: st.write(f"#{g['round']}")
        with c2: st.write(icon)
        with c3: st.write(f"闲: {'-'.join(g['player_cards'])}" if g.get('mode')=='card' else "快速记录")
        with c4: st.write(f"庄: {'-'.join(g['banker_cards'])}" if g.get('mode')=='card' else "快速记录")
        with c5:
            if g['result']=='B': st.error("庄赢")
            elif g['result']=='P': st.info("闲赢")
            else: st.warning("和局")

# ---------------------------- 主程序 ----------------------------
def main():
    tab1,tab2,tab3,tab4 = st.tabs(["🎯 智能分析","🛣️ 六路分析","📊 专业统计","📝 历史记录"])
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
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🔄 开始新牌靴", use_container_width=True):
            st.session_state.ultimate_games.clear()
            st.session_state.expert_roads={'big_road':[], 'bead_road':[], 'big_eye_road':[], 'small_road':[], 'cockroach_road':[], 'three_bead_road':[]}
            st.session_state.risk_data={'current_level':'medium','position_size':1.0,'stop_loss':3,'consecutive_losses':0,'win_streak':0}
            st.session_state.cusum_meter=0.0
            st.success("新牌靴开始！"); st.rerun()
    with c2:
        if st.button("📋 导出数据", use_container_width=True):
            st.info("数据导出功能准备中...")

if __name__ == "__main__":
    main()
