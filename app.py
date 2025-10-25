# -*- coding: utf-8 -*-
# Precision 15 Apex Fusion – Touch+TiePair 终极版
# 基于 Precision 13/14，只加不减：保留全部功能 + Touch 输入 + 和/对子预测面板

import streamlit as st
import numpy as np
import json
from collections import Counter
from datetime import datetime
from itertools import groupby

st.set_page_config(page_title="🐉 百家乐大师 Precision 15 Apex Fusion", layout="centered")

# ============================= 样式 =============================
st.markdown("""
<style>
.main-header {font-size:2.2rem;color:#FFD700;text-align:center;text-shadow:2px 2px 4px #000;}
.prediction-card{background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:15px;border:3px solid #FFD700;margin:15px 0;text-align:center;}
.road-display{background:#1a1a1a;padding:12px;border-radius:8px;margin:8px 0;border:1px solid #333;}
.multi-road{background:#2d3748;padding:10px;border-radius:8px;margin:5px 0;font-family:monospace;}
.risk-panel{background:#2d3748;padding:15px;border-radius:10px;margin:10px 0;border-left:4px solid #e74c3c;}
.metric-table{background:#1f2937;border-radius:10px;padding:10px 12px;margin-top:8px;border:1px solid #334155;color:#e5e7eb;font-size:14px;}
.metric-table .row{display:flex;justify-content:space-between;padding:4px 0;}
.badge{padding:2px 6px;border-radius:6px;font-weight:700;font-size:12px;}
.badge-pos{background:#14532d;color:#bbf7d0;}
.badge-neg{background:#7f1d1d;color:#fecaca;}
.badge-neutral{background:#334155;color:#cbd5e1;}
.state-signal{background:linear-gradient(90deg,#FFD70033,#FF634733);padding:8px 12px;border-radius:8px;margin:5px 0;border-left:4px solid #FFD700;color:#fff;font-weight:600;}
.guide-panel{background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:10px;margin:10px 0;color:white;}
.enhanced-logic-panel{background:linear-gradient(135deg,#00b4db,#0083b0);padding:15px;border-radius:10px;margin:10px 0;color:white;}
.tiepair-panel{background:#0f172a;padding:12px;border-radius:10px;border:1px solid #334155;margin:10px 0;color:#e2e8f0;}
.pill{display:inline-block;padding:4px 10px;border-radius:999px;margin:2px 4px;font-weight:700;background:#1f2937;border:1px solid #334155;}
.pkey{padding:6px 10px;border-radius:8px;border:1px solid #4b5563;background:#111827;color:#e5e7eb;margin:4px;min-width:46px;text-align:center;cursor:pointer;}
.pkey:active{transform:scale(0.98);}
.touch-box{background:#0b1220;padding:12px;border:1px solid #223047;border-radius:10px;}
.touch-title{color:#93c5fd;font-weight:700;margin-bottom:6px;}
.kbd{padding:2px 6px;border:1px solid #475569;border-radius:6px;margin-left:6px;color:#cbd5e1;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐉 百家乐大师 Precision 15 Apex Fusion</h1>', unsafe_allow_html=True)

# ============================= 状态初始化 =============================
def _init_state():
    s = st.session_state
    s.setdefault("ultimate_games", [])
    s.setdefault("expert_roads", {'big_road':[],'bead_road':[],'big_eye_road':[],'small_road':[],'cockroach_road':[],'three_bead_road':[]})
    s.setdefault("risk_data", {'current_level':'medium','position_size':1.0,'stop_loss':3,'consecutive_losses':0,'win_streak':0})
    s.setdefault("ai_weights", {'z':0.25,'cusum':0.25,'bayes':0.20,'momentum':0.15,'eor':0.15})
    s.setdefault("ai_learning_buffer", [])
    s.setdefault("ai_last_metrics", {})
    s.setdefault("ai_entropy", 0.0)
    s.setdefault("eor_decks", 7)
    s.setdefault("ai_batch_n", 5)
    s.setdefault("prediction_stats", {'total_predictions':0,'correct_predictions':0,'recent_accuracy':[],'prediction_history':[]})
    s.setdefault("learning_effectiveness", [])
    s.setdefault("performance_warnings", [])
    s.setdefault("last_prediction", None)
    # Touch 输入临时牌
    s.setdefault("touch_player_cards", [])
    s.setdefault("touch_banker_cards", [])
    s.setdefault("use_touch", True)  # 默认启用 Touch 面板

_init_state()

# ============================= 六路分析 =============================
class CompleteRoadAnalyzer:
    @staticmethod
    def update_all_roads(result):
        if result not in ['B','P']: return
        roads = st.session_state.expert_roads
        roads['bead_road'].append(result)
        if not roads['big_road']:
            roads['big_road'].append([result])
        else:
            col = roads['big_road'][-1]
            if col[-1]==result: col.append(result)
            else: roads['big_road'].append([result])
        # 大眼、小路、蟑螂、三珠
        if len(roads['big_road'])>=2:
            eye=[]; br=roads['big_road']
            for i in range(1,len(br)):
                eye.append('R' if len(br[i])>=len(br[i-1]) else 'B')
            roads['big_eye_road']=eye[-20:]
        if len(roads['big_eye_road'])>=2:
            sm=[]; r=roads['big_eye_road']
            for i in range(1,len(r)): sm.append('R' if r[i]==r[i-1] else 'B')
            roads['small_road']=sm[-15:]
        if len(roads['small_road'])>=2:
            ck=[]; r=roads['small_路'] if False else roads['small_road']  # 兼容保留
            for i in range(1,len(r)): ck.append('R' if r[i]==r[i-1] else 'B')
            roads['cockroach_road']=ck[-12:]
        if len(roads['bead_road'])>=3:
            br=roads['bead_road']; roads['three_bead_road']=[br[i:i+3] for i in range(0,len(br)-2,3)][-8:]

# ============================= 模式检测 =============================
class AdvancedPatternDetector:
    @staticmethod
    def get_streaks(bp):
        if not bp:return []
        s=[];c=bp[0];n=1
        for i in range(1,len(bp)):
            if bp[i]==c:n+=1
            else:s.append(n);c=bp[i];n=1
        s.append(n);return s
    @staticmethod
    def detect_all_patterns(seq):
        bp=[x for x in seq if x in ['B','P']]
        if len(bp)<4:return []
        p=[]; s=AdvancedPatternDetector.get_streaks(bp)
        if len(set(bp[-4:]))==1:p.append(f"{bp[-1]}长龙")
        if len(bp)>=6 and len(set(bp[-6:]))==1:p.append(f"超强{bp[-1]}长龙")
        if len(bp)>=6 and bp[-6:] in [['B','P','B','P','B','P'],['P','B','P','B','P','B']]:p.append("完美单跳")
        if len(s)>=3 and s[-3]==2 and s[-2]==1 and s[-1]==2:p.append("一房一厅")
        if len(s)>=4 and all(s[i]<s[i+1] for i in range(-4,-1)):p.append("上山路")
        return p[:8]

# ============================= 指标计算 =============================
class HybridMathCore:
    @staticmethod
    def compute_metrics(seq):
        bp=[x for x in seq if x in ['B','P']]
        if len(bp)<6:return {'z':0,'cusum':0,'bayes':0,'momentum':0,'entropy':0,'eor':0}
        arr=np.array([1 if x=='B' else -1 for x in bp])
        mean=np.mean(arr); std=np.std(arr)+1e-6
        z=mean/std
        diff=np.diff(arr)
        cusum=np.maximum.accumulate(np.cumsum(diff))[-1]/len(bp)
        bayes=(bp.count('B')+1)/(len(bp)+2)-0.5
        momentum=np.mean(arr[-4:])
        pB=bp.count('B')/len(bp); pP=1-pB
        entropy=-(pB*np.log2(pB+1e-9)+pP*np.log2(pP+1e-9))
        decks=st.session_state.eor_decks
        eor=((pB-pP)*decks)/8
        return {'z':z,'cusum':cusum,'bayes':bayes,'momentum':momentum,'entropy':entropy,'eor':eor}

# ============================= 自学习 =============================
class AIHybridLearner:
    @staticmethod
    def learn_update(correct):
        buf=st.session_state.ai_learning_buffer
        if len(buf)<st.session_state.ai_batch_n:return
        avg={k:np.mean([b[k] for b in buf]) for k in buf[0].keys()}
        w=st.session_state.ai_weights
        for k in w.keys():
            adjust=0.02 if correct else -0.01
            w[k]+=adjust*avg[k]
            w[k]=float(np.clip(w[k],0.05,0.4))
        st.session_state.ai_learning_buffer.clear()
    @staticmethod
    def compute_hybrid(seq):
        m=HybridMathCore.compute_metrics(seq)
        st.session_state.ai_last_metrics=m
        st.session_state.ai_learning_buffer.append(m)
        w=st.session_state.ai_weights
        hybrid=(m['z']*w['z']+m['cusum']*w['cusum']+m['bayes']*w['bayes']+m['momentum']*w['momentum']+m['eor']*w['eor'])
        st.session_state.ai_entropy=m['entropy']
        return hybrid,m

# ============================= 状态信号 =============================
class GameStateDetector:
    @staticmethod
    def _get_current_streak(bead):
        if not bead: return 0
        cur=bead[-1]; n=1
        for i in range(len(bead)-2,-1,-1):
            if bead[i]==cur: n+=1
            else: break
        return n
    @staticmethod
    def _detect_road_breakthrough(big_road):
        if len(big_road)<4:return None
        last4=big_road[-4:]; L=[len(c) for c in last4]
        cur=last4[-1][-1] if last4[-1] else None
        if not cur:return None
        cn="庄" if cur=='B' else "闲"
        if (L[-1]>max(L[-4:-1])+1 and all(x<=2 for x in L[-4:-1])): return f"{cn}势突破"
        if (L[-4]<L[-3]<L[-2]<L[-1]): return f"{cn}势加速"
        return None
    @staticmethod
    def _detect_multi_road_alignment(roads):
        sig=[]
        if roads['big_road'] and roads['big_road'][-1]:
            if len(roads['big_road'][-1])>=3: sig.append(roads['big_road'][-1][-1])
        if roads['big_eye_road']:
            last3=roads['big_eye_road'][-3:]
            if last3 and all(x=='R' for x in last3): sig.append('B')
            elif last3 and all(x=='B' for x in last3): sig.append('P')
        if roads['small_road']:
            last3=roads['small_路'] if False else roads['small_road'][-3:]
            if last3 and len(set(last3))==1: sig.append('B' if last3[0]=='R' else 'P')
        if sig:
            mc=Counter(sig).most_common(1)[0]
            if mc[1]>=2: return "庄趋势" if mc[0]=='B' else "闲趋势"
        return None
    @staticmethod
    def _detect_streak_exhaustion(roads):
        if not roads['bead_road'] or not roads['big_eye_road']: return None
        streak=GameStateDetector._get_current_streak(roads['bead_road'])
        if streak<5: return None
        cur=roads['bead_road'][-1]; cn="庄" if cur=='B' else "闲"
        rev=0
        if len(roads['big_eye_road'])>=2 and roads['big_eye_road'][-1]!=roads['big_eye_road'][-2]: rev+=1
        if len(roads['small_road'])>=3:
            last3=roads['small_road'][-3:]
            if sum(1 for i in range(1,len(last3)) if last3[i]!=last3[i-1])>=2: rev+=1
        if rev>=1: return f"{cn}龙衰竭"
        return None
    @staticmethod
    def detect(roads):
        out=[]
        br=GameStateDetector._detect_road_breakthrough(roads['big_road'])
        if br: out.append(f"大路突破-{br}")
        al=GameStateDetector._detect_multi_road_alignment(roads)
        if al: out.append(f"多路共振-{al}")
        ex=GameStateDetector._detect_streak_exhaustion(roads)
        if ex: out.append(f"连势衰竭-{ex}")
        return out

# ============================= 风险管理 =============================
class ProfessionalRiskManager:
    @staticmethod
    def calculate_position_size(confidence, streak_info):
        base = 1.0
        if confidence > 0.8: base *= 1.2
        elif confidence > 0.7: base *= 1.0
        elif confidence > 0.6: base *= 0.8
        else: base *= 0.5
        if streak_info.get('current_streak', 0) >= 3:
            base *= 1.1
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
        suggestions = {
            "low": {"B": "✅ 庄势明确，可适度加仓",
                    "P": "✅ 闲势明确，可适度加仓",
                    "HOLD": "⚪ 趋势平衡，正常操作"},
            "medium": {"B": "⚠️ 庄势一般，建议轻仓",
                       "P": "⚠️ 闲势一般，建议轻仓",
                       "HOLD": "⚪ 信号不明，建议观望"},
            "high": {"B": "🚨 高波动庄势，谨慎操作",
                     "P": "🚨 高波动闲势，谨慎操作",
                     "HOLD": "⛔ 高风险期，建议休息"},
            "extreme": {"B": "⛔ 极高风险，强烈建议观望",
                        "P": "⛔ 极高风险，强烈建议观望",
                        "HOLD": "⛔ 市场混乱，暂停操作"}
        }
        return suggestions[risk_level].get(direction, "正常操作")

# ============================= 记录/学习 =============================
def record_prediction_result(prediction, actual_result, confidence):
    if actual_result in ['B', 'P']:
        stats = st.session_state.prediction_stats
        stats['total_predictions'] += 1
        is_correct = (prediction == actual_result)
        if is_correct:
            stats['correct_predictions'] += 1
        stats['recent_accuracy'].append(is_correct)
        if len(stats['recent_accuracy']) > 50:
            stats['recent_accuracy'].pop(0)
        stats['prediction_history'].append({
            'prediction': prediction,
            'actual': actual_result,
            'correct': is_correct,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat()
        })

def enhanced_learning_update(prediction, actual_result):
    if prediction in ['B','P'] and actual_result in ['B','P']:
        is_correct = (prediction == actual_result)
        AIHybridLearner.learn_update(correct=is_correct)
        st.session_state.learning_effectiveness.append({
            'correct': is_correct,
            'weights_snapshot': dict(st.session_state.ai_weights),
            'timestamp': datetime.now().isoformat()
        })

# ============================= 看路推荐 =============================
def road_recommendation(roads):
    lines=[]; final=""
    if roads['big_road']:
        last=roads['big_road'][-1]; color_cn="庄" if last[-1]=='B' else "闲"; streak=len(last)
        if streak>=3: lines.append(f"大路：{color_cn}连{streak}局 → 顺路{color_cn}"); final=f"顺大路{color_cn}"
        else: lines.append(f"大路：{color_cn}走势平衡")
    if roads['big_eye_路'] if False else roads['big_eye_road']:
        r=roads['big_eye_road'].count('R'); b=roads['big_eye_road'].count('B')
        if r>b: lines.append("大眼路：红>蓝 → 趋势延续")
        elif b>r: lines.append("大眼路：蓝>红 → 有反转迹象")
        else: lines.append("大眼路：红=蓝 → 稳定期")
    if roads['small_road']:
        r=roads['small_road'].count('R'); b=roads['small_road'].count('B')
        if r>b: lines.append("小路：红>蓝 → 延续趋势")
        elif b>r: lines.append("小路：蓝>红 → 节奏转弱")
        else: lines.append("小路：红=蓝 → 平衡")
    if roads['cockroach_road']:
        last3=roads['cockroach_road'][-3:]
        if last3:
            trend="红红蓝" if last3.count('R')==2 else ("蓝蓝红" if last3.count('B')==2 else "混乱")
            lines.append(f"蟑螂路：{trend} → {'轻微震荡' if trend!='混乱' else '趋势不明'}")
    if not final:
        if roads['big_eye_road']:
            r=roads['big_eye_road'].count('R'); b=roads['big_eye_road'].count('B')
            final="顺路（偏红，延续）" if r>b else ("反路（偏蓝，注意反转）" if b>r else "暂无明显方向")
        else: final="暂无明显方向"
    return {"lines":lines,"final":final}

# ============================= Tie & Pair 概率估计 =============================
def tie_pair_estimator(seq, decks):
    """
    轻量启发式估计（不使用未来信息）：
    - 基线：Tie ~ 9%~10%；PlayerPair/BankerPair ~ 7%~8%（多副牌）
    - 调整项：最近 30 局和局/对子出现频率、熵、高低波动（以 momentum / z / cusum 微调）
    作用：仅显示概率，不参与胜负准确率统计。
    """
    base_tie = 0.095
    base_pp = 0.075
    base_bp = 0.075

    last30 = [x for x in seq[-30:] if x in ['B','P','T']]
    tie_recent = (last30.count('T')/len(last30)) if last30 else base_tie
    bp=[x for x in seq if x in ['B','P']]
    m=st.session_state.ai_last_metrics or {}
    z = abs(m.get('z',0)); cs = abs(m.get('cusum',0)); mom = abs(m.get('momentum',0))
    ent = float(st.session_state.ai_entropy)

    # 熵高 → 偏乱局 → Tie 略增；趋势强(z/cusum高) → Pair 略增
    tie = base_tie*0.7 + tie_recent*0.3 + 0.02*(ent-0.8)
    pp  = base_pp  + 0.01*z + 0.005*cs + 0.002*mom
    bpv = base_bp  + 0.01*z + 0.005*cs + 0.002*mom

    # decks 轻微放大（副数多，组合增多，但影响很小）
    scale = 1.0 + (decks-6)*0.01
    tie *= scale; pp *= scale; bpv *= scale

    # 合理边界
    tie = float(np.clip(tie, 0.05, 0.16))
    pp  = float(np.clip(pp,  0.04, 0.12))
    bpv = float(np.clip(bpv, 0.04, 0.12))
    return tie, pp, bpv

# ============================= 输入与记录 =============================
def parse_cards(input_str):
    if not input_str: return []
    s=input_str.upper().replace(' ',''); cards=[]; i=0
    while i<len(s):
        if i+1<len(s) and s[i:i+2]=='10': cards.append('10'); i+=2
        elif s[i] in '123456789': cards.append(s[i]); i+=1
        elif s[i] in ['A','J','Q','K','0']:
            mp={'A':'A','J':'J','Q':'Q','K':'K','0':'10'}; cards.append(mp[s[i]]); i+=1
        else: i+=1
    return cards

def record_game(result, p_cards, b_cards, mode):
    game={'round':len(st.session_state.ultimate_games)+1,
          'player_cards':p_cards,'banker_cards':b_cards,
          'result':result,'time':datetime.now().strftime("%H:%M"),'mode':mode}
    st.session_state.ultimate_games.append(game)
    if result in ['B','P']: CompleteRoadAnalyzer.update_all_roads(result)
    # 风险状态（只在庄/闲时记为“赢段”，和局不改变连赢/连输）
    risk=st.session_state.risk_data
    if result in ['B','P']:
        risk['win_streak']+=1; risk['consecutive_losses']=0
    else:
        # 和局不重置 streak（避免影响）
        pass
    st.success(f"✅ 记录成功! 第{game['round']}局"); st.rerun()

# ---------- Touch 输入面板 ----------
RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']
def _touch_add(side, rank):
    key = "touch_player_cards" if side=='P' else "touch_banker_cards"
    st.session_state[key].append(rank)

def _touch_clear(side):
    key = "touch_player_cards" if side=='P' else "touch_banker_cards"
    st.session_state[key] = []

def display_touch_input():
    st.markdown("### 📲 Touch 输入（点选 A–K），也可切换键盘输入")
    cA,cB = st.columns(2)
    with cA:
        st.markdown('<div class="touch-box"><div class="touch-title">闲家 (Player)</div>', unsafe_allow_html=True)
        rows=[RANKS[:7], RANKS[7:]]
        for row in rows:
            cols=st.columns(len(row))
            for i,r in enumerate(row):
                if cols[i].button(r, key=f"p_{r}", help="添加到闲家", use_container_width=True):
                    _touch_add('P', r)
        st.caption(f"🃏 当前闲家牌：{'-'.join(st.session_state.touch_player_cards) if st.session_state.touch_player_cards else '—'}")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("清空闲家", key="clrP", use_container_width=True):
                _touch_clear('P')
        with c2:
            pass
        st.markdown('</div>', unsafe_allow_html=True)
    with cB:
        st.markdown('<div class="touch-box"><div class="touch-title">庄家 (Banker)</div>', unsafe_allow_html=True)
        rows=[RANKS[:7], RANKS[7:]]
        for row in rows:
            cols=st.columns(len(row))
            for i,r in enumerate(row):
                if cols[i].button(r, key=f"b_{r}", help="添加到庄家", use_container_width=True):
                    _touch_add('B', r)
        st.caption(f"🃏 当前庄家牌：{'-'.join(st.session_state.touch_banker_cards) if st.session_state.touch_banker_cards else '—'}")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("清空庄家", key="clrB", use_container_width=True):
                _touch_clear('B')
        with c2:
            pass
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("#### 🏆 本局结果")
    r1,r2,r3 = st.columns(3)
    with r1:
        if st.button("🔴 庄赢", use_container_width=True, type="primary"):
            record_game('B', st.session_state.touch_player_cards.copy(), st.session_state.touch_banker_cards.copy(), 'touch')
    with r2:
        if st.button("🔵 闲赢", use_container_width=True):
            record_game('P', st.session_state.touch_player_cards.copy(), st.session_state.touch_banker_cards.copy(), 'touch')
    with r3:
        if st.button("⚪ 和局", use_container_width=True):
            record_game('T', st.session_state.touch_player_cards.copy(), st.session_state.touch_banker_cards.copy(), 'touch')

def handle_card_input(player_input, banker_input, banker_btn, player_btn, tie_btn):
    p=parse_cards(player_input); b=parse_cards(banker_input)
    if len(p)>=2 and len(b)>=2:
        res='B' if banker_btn else ('P' if player_btn else 'T')
        record_game(res,p,b,'card')
    else:
        st.error("❌ 需要至少2张牌")

def handle_quick_input(quick_banker, quick_player):
    res='B' if quick_banker else 'P'
    record_game(res,['X','X'],['X','X'],'quick')

def handle_batch_input(batch_input):
    s=batch_input.upper().replace('庄','B').replace('闲','P').replace('和','T').replace(' ','')
    valid=[c for c in s if c in ['B','P','T']]
    if valid:
        for r in valid: record_game(r,['X','X'],['X','X'],'batch')
        st.success(f"✅ 批量添加{len(valid)}局")

# ============================= 智能分析（含增强逻辑 + Tie/Pair） =============================
def display_complete_analysis():
    if len(st.session_state.ultimate_games)<3:
        st.info("🎲 请先记录至少3局牌局数据"); return

    seq=[g['result'] for g in st.session_state.ultimate_games]
    hybrid, metrics = AIHybridLearner.compute_hybrid(seq)

    with st.sidebar:
        decks = st.slider("EOR 计算副数（1-8）", min_value=1, max_value=8, value=int(st.session_state.eor_decks), key="eor_slider")
        if decks != st.session_state.eor_decks:
            st.session_state.eor_decks = decks
        st.markdown("### 🤖 AI 权重（只读显示）")
        w = st.session_state.ai_weights
        st.write({k: round(v,3) for k,v in w.items()})

    state_signals = GameStateDetector.detect(st.session_state.expert_roads)

    # —— 动态阈值 & 投票 —— #
    st.markdown('<div class="enhanced-logic-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 智能决策引擎")

    ent = float(st.session_state.ai_entropy)
    trend = (abs(st.session_state.ai_last_metrics.get('z',0)) + abs(st.session_state.ai_last_metrics.get('cusum',0))) / 2.0
    thr_base = 0.10 + 0.04*ent - 0.06*trend
    threshold = float(np.clip(thr_base, 0.05, 0.12))

    hist = st.session_state.prediction_stats.get('prediction_history', [])
    hold_adjustment = 1.0
    if len(hist) >= 30:
        hold_ratio = np.mean([1 if h['prediction']=='HOLD' else 0 for h in hist[-30:]])
        if hold_ratio > 0.50:
            threshold *= 0.80
            hold_adjustment = 0.80

    m = metrics
    def sgn(x): return 'B' if x>0 else ('P' if x<0 else 'HOLD')
    votes = [sgn(m['z']), sgn(m['cusum']), sgn(m['momentum']), sgn(m['bayes']), sgn(m['eor'])]
    cnt = Counter([v for v in votes if v!='HOLD'])
    vote_dir, vote_num = (None,0) if not cnt else cnt.most_common(1)[0]

    if hybrid > threshold: prelim = "B"
    elif hybrid < -threshold: prelim = "P"
    else: prelim = "HOLD"

    margin = abs(hybrid) - threshold
    if prelim != "HOLD" and margin < 0.04 and vote_dir in ['B','P'] and vote_dir != prelim:
        direction = vote_dir
        vote_override = True
    else:
        direction = prelim
        vote_override = False

    scale = 0.12
    sigm = 1/(1 + np.exp(-abs(hybrid)/scale))
    base_conf = 0.52 + 0.36*sigm

    if state_signals:
        for sig in state_signals:
            if '突破' in sig or '共振' in sig:
                base_conf = min(0.94, base_conf*1.12)
            if '衰竭' in sig and direction != 'HOLD':
                direction='HOLD'
                base_conf=max(base_conf,0.60)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("动态阈值", f"{threshold:.3f}")
    with c2: st.metric("熵值", f"{ent:.3f}")
    with c3: st.metric("趋势强度", f"{trend:.3f}")
    with c4: st.metric("HOLD调整", f"{hold_adjustment:.2f}")
    if vote_override:
        st.info(f"🎯 投票机制激活：{vote_dir}（{vote_num}/5票）")
    st.markdown('</div>', unsafe_allow_html=True)

    # —— Tie / Pair 概率 —— #
    tie_p, ppair_p, bpair_p = tie_pair_estimator(seq, st.session_state.eor_decks)

    # —— 模式 & 看路 —— #
    patterns = AdvancedPatternDetector.detect_all_patterns(seq)
    road_sug = road_recommendation(st.session_state.expert_roads)

    # —— 预测卡片（含 Tie/Pair 概率条） —— #
    if direction=="B":
        color="#FF6B6B"; icon="🔴"; text="庄(B)"; bg="linear-gradient(135deg,#FF6B6B,#C44569)"
    elif direction=="P":
        color="#4ECDC4"; icon="🔵"; text="闲(P)"; bg="linear-gradient(135deg,#4ECDC4,#44A08D)"
    else:
        color="#FFE66D"; icon="⚪"; text="观望"; bg="linear-gradient(135deg,#FFE66D,#F9A826)"

    vol = float(abs(metrics['momentum']))*0.6 + 0.4*(1 - abs(metrics['bayes']))
    risk_level, risk_text = ProfessionalRiskManager.get_risk_level(base_conf, vol)

    st.markdown(f"""
    <div class="prediction-card" style="background:{bg};">
        <h2 style="color:{color};margin:0;text-align:center;">{icon} 大师推荐: {text}</h2>
        <h3 style="color:#fff;text-align:center;margin:10px 0;">🎯 置信度: {base_conf*100:.1f}% | {risk_text}</h3>
        <p style="color:#f8f9fa;text-align:center;margin:0;">
            模式: {",".join(patterns[:3]) if patterns else "—"} | 风险: {risk_level}
        </p>
        <div class="tiepair-panel">
            <span class="pill">🎲 和局: {(tie_p*100):.1f}%</span>
            <span class="pill">🔴 庄对子: {(bpair_p*100):.1f}%</span>
            <span class="pill">🔵 闲对子: {(ppair_p*100):.1f}%</span>
        </div>
        {"<div style='margin-top:8px;font-weight:700;'>🛣️ 看路推荐：" + road_sug['final'] + "</div>" if (road_sug and road_sug.get('final')) else ""}
    </div>
    """, unsafe_allow_html=True)

    # —— 指标表 —— #
    st.markdown("#### 📐 Hybrid 指标总览")
    def badge(v):
        if v>0: return f'<span class="badge badge-pos">+{v:.3f}</span>'
        if v<0: return f'<span class="badge badge-neg">{v:.3f}</span>'
        return f'<span class="badge badge-neutral">{v:.3f}</span>'
    w = st.session_state.ai_weights
    tbl = f"""
    <div class="metric-table">
      <div class="row"><div>Z-Score</div><div>{badge(metrics['z'])} · w={w['z']:.2f}</div></div>
      <div class="row"><div>CUSUM</div><div>{badge(metrics['cusum'])} · w={w['cusum']:.2f}</div></div>
      <div class="row"><div>Bayes</div><div>{badge(metrics['bayes'])} · w={w['bayes']:.2f}</div></div>
      <div class="row"><div>Momentum</div><div>{badge(metrics['momentum'])} · w={w['momentum']:.2f}</div></div>
      <div class="row"><div>EOR (decks={st.session_state.eor_decks})</div><div>{badge(metrics['eor'])} · w={w['eor']:.2f}</div></div>
      <div class="row"><div>Entropy</div><div>{badge(st.session_state.ai_entropy)}</div></div>
      <div class="row"><div><b>Hybrid 合成</b></div><div><b>{badge(hybrid)}</b></div></div>
      <div class="row"><div>方向</div><div><b>{'庄(B)' if direction=='B' else ('闲(P)' if direction=='P' else '观望')}</b></div></div>
    </div>
    """
    st.markdown(tbl, unsafe_allow_html=True)

    # —— 状态信号 —— #
    if state_signals:
        for ssignal in state_signals:
            st.markdown(f'<div class="state-signal">🚀 状态信号：{ssignal}</div>', unsafe_allow_html=True)

    # —— 风险控制 —— #
    st.markdown("### 🛡️ 风险控制")
    pos = ProfessionalRiskManager.calculate_position_size(base_conf, {'current_streak':0})
    sug = ProfessionalRiskManager.get_trading_suggestion(risk_level, direction)
    st.markdown(f"""
    <div class="risk-panel">
        <h4 style="color:#fff;margin:0 0 10px 0;">📊 风险控制建议</h4>
        <p style="color:#ccc;margin:5px 0;"><strong>仓位建议:</strong> {pos:.1f} 倍基础仓位</p>
        <p style="color:#ccc;margin:5px 0;"><strong>操作建议:</strong> {sug}</p>
        <p style="color:#ccc;margin:5px 0;"><strong>连赢:</strong> {st.session_state.risk_data['win_streak']} 局 | <strong>连输:</strong> {st.session_state.risk_data['consecutive_losses']} 局</p>
    </div>
    """, unsafe_allow_html=True)

    # —— 学习（不使用未来信息；仅在上一局真实结果可用时更新）—— #
    if len(seq) > 1 and st.session_state.last_prediction in ['B','P']:
        last_result = seq[-1]  # 上一条真实结果（已发生）
        record_prediction_result(st.session_state.last_prediction, last_result, base_conf)
        enhanced_learning_update(st.session_state.last_prediction, last_result)

    # 将当前建议存为“上一次预测”，供下一局验证
    st.session_state.last_prediction = direction

# ============================= 界面：输入系统 =============================
def show_quick_start_guide():
    if len(st.session_state.ultimate_games) == 0:
        st.markdown("""
        <div class="guide-panel">
        <h3>🎯 快速开始指南</h3>
        <p>1. 可在「Touch 输入」直接点选 A–K 录入牌，或切换至「键盘输入」方式</p>
        <p>2. 记录 ≥ 3 局后激活 AI 智能分析（不使用未来信息）</p>
        <p>3. 侧栏可调 EOR 副数；权重后台自学习</p>
        <p>4. 和局/对子概率仅显示，不影响主胜负统计</p>
        </div>
        """, unsafe_allow_html=True)

def display_complete_interface():
    st.markdown("## 🎮 双模式输入系统")
    show_quick_start_guide()

    # 输入方式切换
    sw1, sw2 = st.columns(2)
    with sw1:
        if st.button("📲 使用 Touch 输入", use_container_width=True, type="primary"):
            st.session_state.use_touch = True; st.rerun()
    with sw2:
        if st.button("⌨️ 使用键盘输入", use_container_width=True):
            st.session_state.use_touch = False; st.rerun()

    if st.session_state.use_touch:
        display_touch_input()
    else:
        col1,col2=st.columns(2)
        with col1: p_input=st.text_input("闲家牌（例如 K10 或 552）", key="player_card")
        with col2: b_input=st.text_input("庄家牌（例如 55 或 AJ）", key="banker_card")
        st.markdown("### 🏆 本局结果")
        b1,b2,b3=st.columns(3)
        with b1: banker_btn=st.button("🔴 庄赢", use_container_width=True, type="primary")
        with b2: player_btn=st.button("🔵 闲赢", use_container_width=True)
        with b3: tie_btn=st.button("⚪ 和局", use_container_width=True)
        if banker_btn or player_btn or tie_btn:
            handle_card_input(p_input,b_input,banker_btn,player_btn,tie_btn)

        st.info("💡 批量输入（可含和局T）：例如 BPBTBTTBP")
        batch=st.text_input("输入BP/T序列", placeholder="如：BPBTBTTBP", key="batch_input")
        if st.button("✅ 确认批量输入", use_container_width=True) and batch:
            handle_batch_input(batch)

# ============================= 六路 / 统计 / 历史 =============================
def display_complete_roads():
    roads=st.session_state.expert_roads
    st.markdown("## 🛣️ 完整六路分析")
    st.markdown("#### 🟠 珠路 (最近20局)")
    if roads['bead_road']:
        disp=" ".join(["🔴" if x=='B' else "🔵" for x in roads['bead_road'][-20:]])
        st.markdown(f'<div class="road-display">{disp}</div>', unsafe_allow_html=True)
    st.markdown("#### 🔴 大路")
    if roads['big_road']:
        for i,col in enumerate(roads['big_road'][-6:]):
            col_disp=" ".join(["🔴" if x=='B' else "🔵" for x in col])
            st.markdown(f'<div class="multi-road">第{i+1}列: {col_disp}</div>', unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        if roads['big_eye_road']:
            st.markdown("#### 👁️ 大眼路")
            disp=" ".join(["🔴" if x=='R' else "🔵" for x in roads['big_eye_road'][-12:]])
            st.markdown(f'<div class="multi-road">{disp}</div>', unsafe_allow_html=True)
    with c2:
        if roads['small_road']:
            st.markdown("#### 🔵 小路")
            disp=" ".join(["🔴" if x=='R' else "🔵" for x in roads['small_road'][-10:]])
            st.markdown(f'<div class="multi-road">{disp}</div>', unsafe_allow_html=True)
    if roads['three_bead_road']:
        st.markdown("#### 🔶 三珠路")
        for i,g in enumerate(roads['three_bead_road'][-6:]):
            disp=" ".join(["🔴" if x=='B' else "🔵" for x in g])
            st.markdown(f'<div class="multi-road">第{i+1}组: {disp}</div>', unsafe_allow_html=True)

def display_professional_stats():
    if not st.session_state.ultimate_games:
        st.info("暂无统计数据"); return
    games=st.session_state.ultimate_games; results=[g['result'] for g in games]
    bead=st.session_state.expert_roads['bead_road']
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
            if total>0: st.metric("庄胜率", f"{results.count('B')/total*100:.1f}%")
        with d2:
            avg=np.mean([len(list(g)) for k,g in groupby(bead)]) if len(bead)>0 else 0
            st.metric("平均连赢", f"{avg:.1f}局")
        with d3:
            if len(bead)>1:
                changes=sum(1 for i in range(1,len(bead)) if bead[i]!=bead[i-1])
                vol=changes/len(bead)*100
                st.metric("波动率", f"{vol:.1f}%")
    stats = st.session_state.prediction_stats
    if stats['total_predictions'] > 0:
        st.markdown("#### 🎯 AI预测性能")
        col1, col2, col3 = st.columns(3)
        with col1:
            accuracy = (stats['correct_predictions'] / stats['total_predictions']) * 100
            st.metric("总体准确率", f"{accuracy:.1f}%")
        with col2:
            recent_acc = np.mean(stats['recent_accuracy'][-20:]) * 100 if stats['recent_accuracy'] else 0
            st.metric("近期准确率", f"{recent_acc:.1f}%")
        with col3:
            st.metric("总预测数", stats['total_predictions'])

def enhanced_export_data():
    data = {
        'games': st.session_state.ultimate_games,
        'roads': st.session_state.expert_roads,
        'ai_weights': st.session_state.ai_weights,
        'prediction_stats': st.session_state.prediction_stats,
        'export_time': datetime.now().isoformat()
    }
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    st.download_button(
        label="📥 下载完整数据",
        data=json_str,
        file_name=f"baccarat_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

# ============================= 主程序 =============================
def add_system_status_panel():
    with st.sidebar.expander("📊 系统状态", expanded=False):
        total_games = len(st.session_state.ultimate_games)
        st.metric("总局数", total_games)
        stats = st.session_state.prediction_stats
        if stats['total_predictions'] > 0:
            accuracy = (stats['correct_predictions'] / stats['total_predictions']) * 100
            st.metric("预测准确率", f"{accuracy:.1f}%")
            st.metric("总预测数", stats['total_predictions'])
        if total_games > 500:
            st.warning("⚠️ 数据量较大，建议导出数据")
        elif total_games > 200:
            st.info("💾 数据量适中，运行流畅")
        else:
            st.success("✅ 系统运行正常")

def main():
    with st.sidebar:
        st.markdown("## ⚙️ 控制台")
        st.caption("随时调整 EOR 副数；AI 权重后台自动学习，界面只显示不修改。")
        add_system_status_panel()

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
        if not st.session_state.ultimate_games:
            st.info("暂无历史记录")
        else:
            st.markdown("## 📝 完整历史")
            recent=st.session_state.ultimate_games[-12:]
            for g in reversed(recent):
                icon="🃏" if g.get('mode') in ['card','touch'] else ("🎯" if g.get('mode')=='quick' else "📝")
                with st.container():
                    c1,c2,c3,c4,c5=st.columns([1,1,2,2,1])
                    with c1: st.write(f"#{g['round']}")
                    with c2: st.write(icon)
                    with c3: st.write(f"闲: {'-'.join(g['player_cards'])}" if g.get('player_cards') else "—")
                    with c4: st.write(f"庄: {'-'.join(g['banker_cards'])}" if g.get('banker_cards') else "—")
                    with c5:
                        if g['result']=='B': st.error("庄赢")
                        elif g['result']=='P': st.info("闲赢")
                        else: st.warning("和局")

    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        if st.button("🔄 开始新牌靴", use_container_width=True):
            st.session_state.ultimate_games.clear()
            st.session_state.expert_roads={'big_road':[],'bead_road':[],'big_eye_road':[],'small_road':[],'cockroach_road':[],'three_bead_road':[]}
            st.session_state.risk_data={'current_level':'medium','position_size':1.0,'stop_loss':3,'consecutive_losses':0,'win_streak':0}
            st.session_state.prediction_stats={'total_predictions':0,'correct_predictions':0,'recent_accuracy':[],'prediction_history':[]}
            st.session_state.last_prediction=None
            st.session_state.touch_player_cards=[]; st.session_state.touch_banker_cards=[]
            st.success("新牌靴开始！"); st.rerun()
    with c2:
        enhanced_export_data()

if __name__ == "__main__":
    main()
