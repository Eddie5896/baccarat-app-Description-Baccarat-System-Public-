# -*- coding: utf-8 -*-
# Baccarat Master Ultimate - Precision 13.5 Ultimate · EOR Fusion 版
# 界面优化版 - 删除批量输入，优化布局，添加扑克牌按钮

import streamlit as st
import numpy as np
import math
import json
from collections import defaultdict, Counter
from datetime import datetime
from itertools import groupby

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
    
    /* 扑克牌按钮样式 */
    .card-button {
        background: linear-gradient(135deg, #1E293B, #334155);
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 10px;
        margin: 4px;
        color: white;
        font-weight: bold;
        width: 60px;
        height: 50px;
        transition: all 0.2s ease;
    }
    
    .card-button:hover {
        background: linear-gradient(135deg, #00D4FF, #0099CC);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.4);
    }
    
    /* 输入区域样式 */
    .input-section {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* 路单显示样式 */
    .road-display {
        background: #1a1a1a;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border: 1px solid #333;
        font-family: monospace;
    }
    
    .risk-panel {
        background: #2d3748;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #e74c3c;
    }
    
    .metric-table {
        background: #1f2937;
        border-radius: 10px;
        padding: 10px 12px;
        margin-top: 8px;
        border: 1px solid #334155;
        color: #e5e7eb;
        font-size: 14px;
    }
    
    .metric-table .row {
        display: flex;
        justify-content: space-between;
        padding: 4px 0;
    }
    
    .badge {
        padding: 2px 6px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
    
    .badge-pos {
        background: #14532d;
        color: #bbf7d0;
    }
    
    .badge-neg {
        background: #7f1d1d;
        color: #fecaca;
    }
    
    .badge-neutral {
        background: #334155;
        color: #cbd5e1;
    }
    
    .state-signal {
        background: linear-gradient(90deg, #FFD70033, #FF634733);
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #FFD700;
        color: #fff;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐉 百家乐大师 Precision 13.5 · EOR Fusion</h1>', unsafe_allow_html=True)

# ========================== 状态初始化 ==========================
def _init_state():
    ss = st.session_state
    ss.setdefault("ultimate_games", [])
    ss.setdefault("expert_roads", {
        'big_road': [],
        'bead_road': [],
        'big_eye_road': [],
        'small_road': [],
        'cockroach_road': [],
        'three_bead_road': []
    })
    ss.setdefault("risk_data", {
        'current_level': 'medium',
        'position_size': 1.0,
        'stop_loss': 3,
        'consecutive_losses': 0,
        'win_streak': 0
    })
    ss.setdefault("ai_weights", {
        'z': 0.25,
        'cusum': 0.25,
        'bayes': 0.20,
        'momentum': 0.15,
        'eor': 0.15
    })
    ss.setdefault("ai_learning_buffer", [])
    ss.setdefault("ai_last_metrics", {})
    ss.setdefault("ai_entropy", 0.0)
    ss.setdefault("eor_decks", 7)
    ss.setdefault("ai_batch_n", 5)
    ss.setdefault("prediction_stats", {
        'total_predictions': 0,
        'correct_predictions': 0,
        'recent_accuracy': [],
        'prediction_history': []
    })
    ss.setdefault("learning_effectiveness", [])
    ss.setdefault("performance_warnings", [])
    ss.setdefault("last_prediction", None)
    ss.setdefault("weight_performance", {
        'z': [],
        'cusum': [],
        'bayes': [],
        'momentum': [],
        'eor': []
    })
    # 13.5 新增：HOLD 目标上限
    ss.setdefault("hold_cap_ratio", 0.15)  # HOLD 不超过 15%
    
    # 新增：扑克牌输入状态
    ss.setdefault("player_cards_input", "")
    ss.setdefault("banker_cards_input", "")
_init_state()

# ========================== 六路分析（保留） ==========================
class CompleteRoadAnalyzer:
    @staticmethod
    def update_all_roads(result):
        if result not in ['B','P']: 
            return
        roads = st.session_state.expert_roads
        roads['bead_road'].append(result)
        if not roads['big_road']:
            roads['big_road'].append([result])
        else:
            col = roads['big_road'][-1]
            if col[-1]==result: col.append(result)
            else: roads['big_road'].append([result])
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
            ck=[]; r=roads['small_road']
            for i in range(1,len(r)): ck.append('R' if r[i]==r[i-1] else 'B')
            roads['cockroach_road']=ck[-12:]
        if len(roads['bead_road'])>=3:
            br=roads['bead_road']; roads['three_bead_road']=[br[i:i+3] for i in range(0,len(br)-2,3)][-8:]

# ========================== 模式检测（保留） ==========================
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

# ========================== GameState（保留） ==========================
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
        if roads['big_路'] if False else roads['big_road']:
            if roads['big_road'] and roads['big_road'][-1] and len(roads['big_road'][-1])>=3:
                sig.append(roads['big_road'][-1][-1])
        if roads['big_eye_road']:
            last3=roads['big_eye_road'][-3:]
            if last3 and all(x=='R' for x in last3): sig.append('B')
            elif last3 and all(x=='B' for x in last3): sig.append('P')
        if roads['small_road']:
            last3=roads['small_road'][-3:]
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

# ========================== EOR / 指标核心（保留） ==========================
class HybridMathCore:
    @staticmethod
    def _eor_plus(seq, roads, decks):
        bp=[x for x in seq if x in ['B','P']]
        n=len(bp)
        if n<6:
            return 0.0

        def bias_win(k):
            if n<k: return 0.0
            last=bp[-k:]
            pB=last.count('B')/k
            return (pB - (1-pB))

        win12 = bias_win(12)
        win24 = bias_win(24)
        win48 = bias_win(48)
        fused_bias = (0.50*win12 + 0.30*win24 + 0.20*win48)

        align = 0.0
        if roads['big_road'] and roads['big_road'][-1]:
            last_col = roads['big_road'][-1]
            cur = last_col[-1]
            if len(last_col)>=3: align += 0.08 if cur=='B' else -0.08
        if roads['big_eye_road']:
            last3=roads['big_eye_road'][-3:]
            if last3 and all(x=='R' for x in last3): align += 0.06
            elif last3 and all(x=='B' for x in last3): align -= 0.06
        if roads['small_road']:
            last3=roads['small_road'][-3:]
            if last3 and len(set(last3))==1:
                align += 0.05 if last3[0]=='R' else -0.05

        pB = bp.count('B')/n
        pP = 1-pB
        entropy = -(pB*np.log2(pB+1e-9)+pP*np.log2(pP+1e-9))
        entropy_penalty = (1.0 - 0.35*entropy)

        deck_scale = np.sqrt(max(1, decks))/4.0

        raw = (fused_bias * 0.85 + align) * entropy_penalty
        eor_plus = float(np.clip(raw * (1.0 + deck_scale), -0.6, 0.6))
        return eor_plus

    @staticmethod
    def compute_metrics(seq):
        bp=[x for x in seq if x in ['B','P']]
        if len(bp)<6:
            m = {'z':0.0,'cusum':0.0,'bayes':0.0,'momentum':0.0,'entropy':0.0,'eor':0.0}
            st.session_state.ai_last_metrics = m
            st.session_state.ai_entropy = 0.0
            return m
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
        roads = st.session_state.expert_roads
        eor = HybridMathCore._eor_plus(seq, roads, decks)

        m = {'z':float(z),'cusum':float(cusum),'bayes':float(bayes),'momentum':float(momentum),'entropy':float(entropy),'eor':float(eor)}
        st.session_state.ai_last_metrics = m
        st.session_state.ai_entropy = entropy
        return m

# ========================== 自学习（保留） ==========================
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
        buf.clear()
    @staticmethod
    def compute_hybrid(seq):
        m=HybridMathCore.compute_metrics(seq)
        st.session_state.ai_learning_buffer.append(m)
        w=st.session_state.ai_weights
        hybrid=(m['z']*w['z']+m['cusum']*w['cusum']+m['bayes']*w['bayes']+m['momentum']*w['momentum']+m['eor']*w['eor'])
        return float(hybrid), m

# ========================== 权重自适应 / 多时间框架 / 风险（保留） ==========================
class EnhancedLogicCore:
    @staticmethod
    def enhanced_dynamic_threshold(seq, metrics, roads):
        ent = st.session_state.ai_entropy
        trend = (abs(metrics['z']) + abs(metrics['cusum'])) / 2.0
        thr_base = 0.10 + 0.04*ent - 0.06*trend
        sample_adjust = max(0, 1 - len(seq) / 100) * 0.03
        thr_base += sample_adjust
        patterns = AdvancedPatternDetector.detect_all_patterns(seq)
        pattern_strength = len(patterns) * 0.01
        thr_base += min(pattern_strength, 0.05)
        road_alignment = EnhancedLogicCore.calculate_road_alignment(roads)
        thr_base -= road_alignment * 0.02
        return float(np.clip(thr_base, 0.04, 0.12))

    @staticmethod
    def calculate_road_alignment(roads):
        alignment_score = 0.0
        if roads['big_road'] and roads['big_road'][-1]:
            current_trend = roads['big_road'][-1][-1]
            if roads['big_eye_road']:
                big_eye_trend = 'B' if roads['big_eye_road'][-3:].count('R') >= 2 else 'P'
                if big_eye_trend == current_trend: alignment_score += 0.3
            if roads['small_road']:
                small_trend = 'B' if roads['small_road'][-3:].count('R') >= 2 else 'P'
                if small_trend == current_trend: alignment_score += 0.2
        return min(alignment_score, 1.0)

    @staticmethod
    def adaptive_weight_optimization(seq, actual_results):
        if len(actual_results) < 20:
            return st.session_state.ai_weights
        recent_games = min(30, len(actual_results))
        metric_performance = {}
        for metric_name in ['z', 'cusum', 'bayes', 'momentum', 'eor']:
            correct_predictions = 0
            total_predictions = 0
            for i in range(len(seq)-recent_games, len(seq)):
                if i <= 0: continue
                metric_value = HybridMathCore.compute_metrics(seq[:i])[metric_name]
                predicted = 'B' if metric_value > 0.05 else ('P' if metric_value < -0.05 else 'HOLD')
                if predicted != 'HOLD' and i < len(actual_results) and predicted == actual_results[i]:
                    correct_predictions += 1
                if predicted != 'HOLD': total_predictions += 1
            if total_predictions > 0:
                metric_performance[metric_name] = correct_predictions / total_predictions
                st.session_state.weight_performance[metric_name].append(metric_performance[metric_name])
                if len(st.session_state.weight_performance[metric_name]) > 50:
                    st.session_state.weight_performance[metric_name].pop(0)
            else:
                metric_performance[metric_name] = 0.5
        total_perf = sum(metric_performance.values())
        if total_perf > 0:
            new_w = {}
            for k, perf in metric_performance.items():
                new_w[k] = perf / total_perf * 0.8 + 0.04
            for k in new_w:
                st.session_state.ai_weights[k] = 0.7*st.session_state.ai_weights[k] + 0.3*new_w[k]
        return st.session_state.ai_weights

    @staticmethod
    def multi_timeframe_confirmation(seq, current_direction, current_confidence):
        if len(seq) < 15:
            return current_direction, current_confidence
        short_term = seq[-8:]
        short_metrics = HybridMathCore.compute_metrics(short_term)
        short_hybrid = sum(short_metrics[k] * st.session_state.ai_weights[k] for k in st.session_state.ai_weights)
        mid_term = seq[-15:]
        mid_metrics = HybridMathCore.compute_metrics(mid_term)
        mid_hybrid = sum(mid_metrics[k] * st.session_state.ai_weights[k] for k in st.session_state.ai_weights)
        short_dir = 'B' if short_hybrid > 0.08 else ('P' if short_hybrid < -0.08 else 'HOLD')
        mid_dir = 'B' if mid_hybrid > 0.06 else ('P' if mid_hybrid < -0.06 else 'HOLD')
        if current_direction != 'HOLD' and short_dir == mid_dir == current_direction:
            enhanced_conf = min(0.95, current_confidence * 1.15)
            return current_direction, enhanced_conf
        elif current_direction != 'HOLD' and (short_dir != current_direction or mid_dir != current_direction):
            reduced = current_confidence * 0.7
            if reduced < 0.55:
                return 'HOLD', max(0.6, reduced)
            else:
                return current_direction, reduced
        return current_direction, current_confidence

    @staticmethod
    def quantify_pattern_strength(patterns, roads):
        strength = 0.0
        for pattern in patterns:
            if '长龙' in pattern:
                strength += 0.15 if '超强' in pattern else 0.08
            elif '完美单跳' in pattern:
                strength += 0.12
            elif '一房一厅' in pattern or '上山路' in pattern:
                strength += 0.06
        for signal in GameStateDetector.detect(roads):
            if '突破' in signal: strength += 0.10
            elif '共振' in signal: strength += 0.07
        return min(strength, 0.3)

    @staticmethod
    def risk_aware_position_sizing(confidence, direction, metrics, consecutive_wins):
        base = 1.0
        if confidence > 0.8: base *= 1.2
        elif confidence > 0.7: base *= 1.0
        elif confidence > 0.6: base *= 0.8
        else: base *= 0.5
        volatility = metrics['entropy'] + abs(metrics['z']) * 0.5
        base *= (1.0 - min(volatility, 0.5))
        if consecutive_wins >= 3:
            base *= min(1.2, 1.0 + consecutive_wins * 0.05)
        patterns = AdvancedPatternDetector.detect_all_patterns([g['result'] for g in st.session_state.ultimate_games])
        base *= (1.0 + EnhancedLogicCore.quantify_pattern_strength(patterns, st.session_state.expert_roads))
        return float(min(base, 2.0))

# ========================== 风控字典（保留） ==========================
class ProfessionalRiskManager:
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
            "low": {"B": "✅ 庄势明确，可适度加仓","P": "✅ 闲势明确，可适度加仓","HOLD": "⚪ 趋势平衡，正常操作"},
            "medium": {"B": "⚠️ 庄势一般，建议轻仓","P": "⚠️ 闲势一般，建议轻仓","HOLD": "⚪ 信号不明，建议观望"},
            "high": {"B": "🚨 高波动庄势，谨慎操作","P": "🚨 高波动闲势，谨慎操作","HOLD": "⛔ 高风险期，建议休息"},
            "extreme": {"B": "⛔ 极高风险，强烈建议观望","P": "⛔ 极高风险，强烈建议观望","HOLD": "⛔ 市场混乱，暂停操作"}
        }
        return suggestions[risk_level].get(direction, "正常操作")

# ========================== 看路推荐（保留） ==========================
def road_recommendation(roads):
    lines=[]; final=""
    if roads['big_road']:
        last=roads['big_road'][-1]; color_cn="庄" if last[-1]=='B' else "闲"; streak=len(last)
        if streak>=3: lines.append(f"大路：{color_cn}连{streak}局 → 顺路{color_cn}"); final=f"顺大路{color_cn}"
        else: lines.append(f"大路：{color_cn}走势平衡")
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
    if roads['cockroach_road']:
        last3=roads['cockroach_road'][-3:]; 
        if last3:
            trend="红红蓝" if last3.count('R')')==2 else ("蓝==2 else ("蓝蓝红" if last3.count蓝红" if last3.count('('B')==2B')==2 else "混乱")
            lines.append(f"蟑螂路：{trend} → else "混乱")
            lines.append(f"蟑螂路：{ {'轻微震荡' if trend!='混乱trend} → {'轻微震荡' if trend!='混乱' else '趋势' else '趋势不明'}")
    if not final:
不明'}")
    if not final:
        if roads['big        if roads['big__eye_路'] if Falseeye_路'] if False else else roads['big_eye roads['big_eye_road']:
_road']:
            r=roads['            r=roads['big_eye_road'].count('R');big_eye_road'].count('R'); b=roads['big_eye_road b=roads['big_eye_road'].count('B'].count('B')
           ')
            final="顺路 final="顺路（偏（偏红，延续）红，延续）" if" if r>b else (" r>b else ("反路（偏蓝，注意反转反路（偏蓝，注意反转）" if b>）" if b>rr else "暂无明显方向")
 else "暂无明显方向")
               else: final="暂 else: final="暂无明显方向无明显方向"
    return {""
    return {"lines":lines,"final":final}

lines":lines,"final":final}

# ========================== 辅助# ========================== 辅助输入/记录（保留）输入/记录（保留） ========================= ==========================
def=
def parse_cards(input parse_cards(input_str):
    if not_str):
    if not input_str input_str: return []
    s: return []
    s=input_str=input_str.upper().replace.upper().replace(' ','(' ',''); cards=[]'); cards=[]; i; i=0
   =0
    while i while i<len(s):
<len(s):
        if        if i+1<len i+1<len(s)(s) and s[i and s[i:i+2]==':i+2]=='10': cards.append10': cards.append('10'); i+=2
        elif s[i('10'); i+=2
        elif s[i] in '] in '123456789': cards.append123456789': cards.append(s[i(s[i]); i+=1]); i+=1
        elif s[i] in ['
        elif s[i] in ['A','J','QA','J','Q','','K','0']:
           K','0']:
            mp={' mp={'A':'A','J':'J','Q':'Q','K':'K','0A':'A','J':'J','Q':'Q','K':'K','0':'10'}; cards':'10'}; cards.append(m.append(mp[s[i]]);p[s[i]]); i i+=1
        else+=1
        else:: i+=1
    i+=1
    return cards

def record_game(result, return cards

def record_game(result, p_cards, b p_cards, b_cards_cards, mode):
   , mode):
    game={'round':len(st.session_state.ultimate_games)+1,
          'player_cards':p game={'round':len(st.session_state.ultimate_games)+1,
          'player_cards':p_cards,'_cards,'banker_cbanker_cards':b_cards,
ards':b_cards,
          '          'resultresult':result,'time':datetime.now().strftime':result,'time':datetime.now().strftime("%("%H:%M"),'modeH:%M"),'mode':mode}
   ':mode}
    st.session_state. st.session_state.ultimate_games.append(game)
    if result inultimate_games.append(game)
    if result in ['B','P']: Complete ['B','P']: CompleteRoadAnalyzer.update_all_roadsRoadAnalyzer.update_all_roads(result)
    risk=st(result)
    risk=st.session_state.risk_data
.session_state.risk_data
    if result in ['B    if result in ['B','P']: 
        risk['','P']: 
        risk['win_streak']+=1; risk['consecutive_losses']=0win_streak']+=1; risk['consecutive_losses']=0
    elif result=='
    elif result=='T':
        pass
    else:
T':
        pass
    else        risk['consecutive_loss:
        risk['conseces']+=1; riskutive_losses']+=1['win_streak']; risk['win_streak']=0
    st.success=0
    st.success(f(f"✅ 记录成功! "✅ 记录成功! 第{game['round']第{game['round']}局"); st.rer}局"); st.rerun()

def handle_card_inputun()

def handle_card_input(player_input, banker_input, banker(player_input, banker_input, banker__btn, player_btn,btn, player_btn, tie tie_btn):
    p_btn):
    p==parse_cards(player_inputparse_cards(player_input); b); b=parse_cards=parse_cards(banker(banker_input)
    if_input)
    if len(p)>=2 and len(b len(p)>=2 and len(b)>=2:
        res)>=2:
        res='B' if banker_='B' if banker_btn elsebtn else ('P' if ('P' if player_ player_btn else 'T')
       btn else 'T')
        record record_game(res,p,b,'_game(res,p,b,'card')
    else:
        st.errorcard')
    else:
        st.error("❌ 需要至少("❌ 需要至少2张牌")

2张牌")

def handle_quick_input(quick_banker, quick_player):
   def handle_quick_input(quick_banker, quick_player):
    res='B' if quick_banker res='B' if quick_banker else 'P'
    record else 'P'
    record_game(res,['X','_game(res,['X','X'],['X','X'],X'],['X','X'quick')

# ='],'quick')

# ========================== 扑克牌按钮输入========================= 扑克牌按钮输入功能 ==========================
def card功能 ==========================
def card_button_interface():
    """_button_interface():
    """显示扑克牌按钮选择显示扑克牌按钮选择界面"""
    st.markdown("界面"""
    st.markdown("###### 🃏 扑 🃏 扑克克牌选择")
    
   牌选择")
    
    # # 定义扑 定义扑克克牌
    cards牌
    cards = = ['A', '2 ['A', '2', '3', '4', '5', '6', '3', '4', '5', '6', '7', '8', '9', '10', '7', '8', '9', '10', '', 'J', 'Q',J', 'Q', 'K 'K']
    
    # ']
    
    # 闲家闲家牌输入区域
牌输入区域
    col1, col2 = st    col1, col2.columns(2)
    
    = st.columns(2)
    
 with col1:
        st.mark    with col1:
       down("#### 🔵 闲 st.markdown("#### 🔵 闲家牌")
       家牌")
        player_input = st.text_input(
            player_input = st.text_input(
            "闲家牌 (手动输入)", 
            value=st.session "闲家牌 (手动输入_state.player_cards_input,
)", 
            value=st.session_state.player_cards_input,
            placeholder="例如: A            placeholder="例如: A10 或 552",
10 或 552",
            key="player_input"
        )
            key="player_input"
        )
        st.session_state.player_c        st.session_state.player_cards_input = player_input
        
ards_input = player_input
        
        # 闲家扑        # 闲家扑克牌按钮
        st.markdown克牌按钮
        st.markdown("**点击添加牌面("**点击添加牌面:**")
        cols = st:**")
        cols = st.columns(4)
.columns(4)
        for i, card in enumerate(cards        for i, card in enumerate(cards):
            with cols[i % ):
            with cols[i % 4]:
                if st.button4]:
                if st.button(f"♠{card(f"♠{card}", key=f"p_{card}", key=f"p_{card}"):
                    st.session_state.player}"):
                    st.session_state.player_cards_input += card
_cards_input += card
                    st.rerun()
                    st.rerun()
        
        # 闲家特殊        
        # 闲家特殊功能按钮
        col_p1功能按钮
        col_p1, col_p2 = st, col_p2 = st.columns(2)
        with.columns(2)
        with col_p1:
            if st col_p1:
            if st.button("清空闲家",.button("清空闲家", key key="clear_player"):
                st.session_state.player_cards="clear_player"):
                st.session_state.player_cards_input = ""
                st.rerun()
        with col_p2_input = ""
                st.rerun()
        with col_p2:
            if:
            if st.button("删除 st.button("删除最后", key="backspace_player"):
               最后", key="backspace_player"):
                if st.session if st.session_state.player_cards_state.player_cards_input:
_input:
                    st                    st.session_state.player_cards.session_state.player_cards_input_input = st = st.session_state.player.session_state.player_cards_cards_input[:-_input[:-1]
               1]
                st.rerun()
    
 st.rerun()
    
    with col2:
        st.markdown    with col2:
        st.markdown("#### 🔴 庄家("#### 🔴 庄家牌")
        banker_input = st牌")
        banker_input = st.text.text_input(
            "_input(
            "庄家牌庄家牌 (手动输入 (手动输入)", 
            value)", 
            value=st.session_state.banker_cards_input,
            placeholder=st.session_state.banker_cards_input,
            placeholder="例如="例如: 55 : 55 或或 AJ",
            key=" AJ",
            key="bankerbanker_input"
        )
_input"
        )
        st        st.session_state.banker_cards.session_state.banker_cards_input = banker_input
        
       _input = banker_input
        
        # # 庄家扑克 庄家扑克牌按钮牌按钮
        st.mark
        st.markdown("**点击down("**点击添加牌面:**")
        cols = st.columns(4)
       添加牌面:**")
        cols = st.columns(4)
        for i for i, card in enumerate, card in enumerate(cards(cards):
            with cols):
            with cols[i %[i % 4]:
                if st 4]:
                if st.button.button(f"♥{card(f"♥{card}",}", key=f"b_{ key=f"b_{card}"card}"):
                    st.session_state):
                    st.session_state.b.banker_cards_input += cardanker_cards_input += card
                    st.rerun
                    st.rerun()
()
        
        # 庄        
        # 庄家特殊家特殊功能按钮
       功能按钮
        col_b col_b1, col_b2 = st.columns(2)
       1, col_b2 = st.columns(2)
        with col_b1:
 with col_b1:
            if            if st.button("清空 st.button("清空庄家庄家", key="clear_b", key="clear_bankeranker"):
                st.session_state.b"):
                st.session_state.banker_cards_input = ""
anker_cards_input = ""
                st.rerun()
                st.rerun()
        with        with col_b col_b2:
            if st.button("删除最后",2:
            if st.button("删除最后", key="backspace_banker"):
                if st.session_state key="backspace_banker"):
                if st.session_state.banker.banker_cards_input:
                   _cards_input:
                    st.session_state.banker_cards_input = st.session_state.banker st.session_state.banker_cards_input =_c st.session_state.banker_cards_inputards_input[:-1[:-1]
                st.r]
                st.rerun()
    
erun()
    
    return player_input    return player_input, banker_input

, banker_input

# =# ========================== 系统========================= 系统面板 /面板 / 导出（保留 导出（保留） ==========================
def add_system_status_p） ==========================
def add_system_status_panel():
    withanel():
    with st st.sidebar.sidebar.expander.expander("("📊📊 系统状态 系统状态", expanded=False):
        total", expanded=False):
        total_games = len(st.session_state_games = len(st.session_state.ultimate_g.ultimate_games)
        stames)
        st.metric.metric("总局数",("总局数", total_games total_games)
        stats)
        stats = st.session_state = st.session_state.prediction.prediction_stats
       _stats
        if stats['total if stats['total_predict_predictions'] > ions'] > 0:
           0:
            accuracy = ( accuracy = (stats['correct_pstats['correct_predictions']redictions'] / stats[' / stats['total_predictionstotal_predictions']) *']) * 100
            100
            st.metric st.metric("预测准确("预测准确率", f"{accuracy:.1f}%率", f"{accuracy:.1f}%")
            st.m")
            st.metric("总预测数", stats['totaletric("总预测数", stats_predictions'])
        if total['total_predictions'])
        if total_games > 500_games > 500:
:
            st.warning("⚠️ 数据量较大，            st.warning("⚠️ 数据量较大，建议导出数据")
        elif建议导出数据")
        elif total_games > 200 total_games > 200:
            st.info("💾:
            st.info(" 数据量适中，运行流畅💾 数据量适中，运行")
        else:
            st流畅")
        else:
            st.success("✅ 系统运行.success("✅ 系统运行正常")

def enhanced_export正常")

def enhanced_export_data():
    data = {
        '_data():
    data = {
games': st.session_state.        'games': st.session_state.ultimate_games,
       ultimate_games,
        ' 'roads': st.session_state.expert_roads,
        'roads': st.session_state.expert_roads,
        'ai_weights': st.session_state.ai_weights,
       ai_weights': st.session_state.ai_weights,
        'prediction_stats': st.session_state 'prediction_stats': st.session_state.prediction_stats,
.prediction_stats,
        'weight_performance': st        'weight_performance': st.session_state.weight_performance,
.session_state.weight_performance,
        'export_time': datetime        'export_time': datetime.now().isoformat()
   .now().isoformat()
    }
    json_str = json.dumps(data, ensure_ascii }
    json_str = json.dumps(data, ensure_ascii=False,=False, indent=2)
 indent=2)
    st.download    st.download_button(
_button(
        label="        label="📥 下载📥 下载完整数据",
完整数据",
        data=json        data=json_str,
_str,
        file        file_name=f"baccarat_name=f"baccarat_data_{_data_{datetime.now().strdatetime.now().strftime('%Yftime('%Y%m%m%d%d__%%HH%M')}.%M')}.json",
        mime="application/jsonjson",
        mime="application/json"
    )

def show_quick_start"
    )

def show_quick_start_guide():
    if len(st.session_guide():
    if len(st.session_state.ultimate_games) == _state.ultimate_games) == 0:
        st.markdown("0:
        st.markdown("""
""
        <div style="background:linear-gradient(        <div style="background:linear-gradient(135deg,#667135deg,#667eeaeea,#764ba2,#764ba2);padding:20);padding:20px;px;border-radius:10border-radius:10px;px;margin:10pxmargin:10px 0;color 0;color:white:white;">
        <h;">
        <h3>🎯3>🎯 快速 快速开始指南</h3>
        <开始指南</h3>
        <p>1. 选择p>1. 选择「牌「牌点输入」记录详细点输入」记录详细牌牌局，或使用「快速局，或使用「快速看路」快速开始</p>
        <p>2看路」快速开始</p>
        <p>2. 记录3. 记录3局后局后激活AI智能分析系统激活AI智能分析系统</p</p>
        <p>
        <p>3. 关注风险建议，科学>3. 关注风险建议，科学管理仓位</管理仓位</p>
        <p>4.p>
        <p>4. 系统 系统会持续会持续学习优化预测准确性学习优化预测准确性</p>
       </p>
        </div </div>
        """, unsafe>
        """, unsafe_allow_allow_html=True)

# =_html=True)

# ========================== 统计记录（========================= 统计记录（保留）保留） ==========================
def record_prediction_result(pred ==========================
def record_prediction_result(prediction, actual_result, confidence):
    ifiction, actual_result, confidence):
    if actual_result in ['B actual_result in ['B', '', 'P']:
       P']:
        stats = st.session_state.prediction_stats
        stats['total_predict stats = st.session_state.prediction_stats
        stats['total_predictions'] +=ions'] +=  1
        is_correct1
        is_correct = (prediction == actual_result)
 = (prediction == actual_result)
        if is_correct:
        if is_correct:
                       stats['correct_predictions stats['correct_predictions']'] += 1
        stats += 1
        stats['recent_accuracy'].['recent_accuracy'].append(is_correct)
       append(is_correct)
        if if len(stats['recent_ len(stats['recent_accuracy']) > 50:
accuracy']) > 50:
            stats['recent_            stats['recent_accuracy'].pop(0)
accuracy'].pop(0)
               stats['prediction_history'].append stats['prediction_history'].append({
            '({
            'predictionprediction': prediction,
            'actual': actual_result,
': prediction,
            'actual': actual_result,
            'correct': is_correct,
            '            'correct': is_correct,
            'confidenceconfidence': float(confidence': float(confidence),
            'timestamp': datetime.now().),
            'timestamp': datetime.now().isoformat()
        })

defisoformat()
        })

def enhanced_learning_update(prediction enhanced_learning_update(prediction, actual_result):
    if prediction, actual_result):
    if prediction in ['B','P'] in ['B','P'] and actual_result in ['B and actual_result in ['B','P']:
        is','P']:
        is_correct = (prediction == actual_correct = (prediction == actual_result)
        AIHybrid_result)
        AIHybridLearner.learn_updateLearner.learn_update(correct=(correct=is_correct)
is_correct)
        st.session        st.session_state.learning__state.learningeffectiveness.append({
            'correct_effectiveness.append({
            '': bool(is_correct),
correct': bool(is_correct),
            'weights_snapshot            'weights_snapshot': dict(st.session_state.ai': dict(st.session_state.ai_weights),
            'timestamp_weights),
            'timestamp':': datetime.now().isoformat datetime.now().isoformat()
        })

# ========================== ()
        })

# ========================== 智能分析（保留） ==========================
智能分析（保留） ==========================
def display_completedef display_complete_analysis_analysis():
    if len(st():
    if len(st.session_state.ultimate_games.session_state.ultimate_games)<3:
        st)<3:
        st.info("🎲.info("🎲 请先记录 请先记录至少3至少3局牌局数据");局牌局数据"); return

    seq=[g[' return

    seq=[gresult'] for g in st.session['result'] for g in st.session_state.ultimate_games_state.ultimate_games]
    hybrid, metrics = AIHy]
    hybrid,bridLearner.compute_ metrics = AIHybridLearner.computehybrid(seq_hybrid(seq)

    with st.side)

    with st.sidebar:
bar:
        decks = st        decks.slider = st.slider("EOR 计算副数（1("EOR 计算副数（1-8）", 1, 8, int(st.session_state-8）", 1, 8, int(st.session_state.eor_decks), key="eor_slider")
       .eor_decks), key="eor_slider")
 if decks != st.session        if decks != st.session_state.eor_de_state.eorcks:
            st.session_state.eor_decks_decks:
            st.session_state.eor_decks = decks
        st.mark = decks
        st.markdown("### 🤖down("### 🤖 AI 权重（ AI 权重（动态优化后）")
        st.write动态优化后）")
        st.write({k: round(v,3) for({k: round(v,3) for k,v in st.session_state k,v in st.session_state.ai_weights.items.ai_weights.items()})

    state_signals = GameStateDet()})

    state_signals = GameStateDetector.detect(st.session_stateector.detect(st.session_state.expert_roads.expert_roads)

    st.markdown("""
)

    st.markdown("""
    <div style="background:linear    <div style="background:linear-gradient(135deg,#00-gradient(135deg,#00b4b4db,#0083b0);padding:15db,#0083b0);padding:15px;border-radius:10pxpx;border-radius:10px;margin:10px ;margin:10px 00;color:white;">
;color:white;">
    """, unsafe_allow_html=True)
    """, unsafe_allow_html=True)
    st.markdown("### 🧠 智能    st.markdown("### 🧠 智能决策引擎决策引擎（EOR（EOR+ + + + 动态阈值 + 动态阈值 + 限频H限频HOLD）")

   OLD）")

    # # 动态阈值
    threshold 动态阈值
    threshold = Enhanced = EnhancedLogicCoreLogicCore.enhanced_d.enhanced_dynamic_threshold(seqynamic_threshold(seq, metrics, metrics, st.session_state, st.session_state.expert.expert_roads_roads)

    #)

    # 权重自适应
    actual_results 权重自适应
    actual_results = = [g['result'] [g['result'] for g for g in st.session in st.session_state.ultimate_state.ultimate_games]
   _games]
    optimized_weights optimized_weights = EnhancedLogicCore. = EnhancedLogicCore.adaptive_weightadaptive_weight_optimization(_optimization(seq,seq, actual_results)

    # 用优化后的权重 actual_results)

    # 用优化后的权重修正 hybrid
修正 hybrid
    hybrid =    hybrid = (metrics (metrics['['z'] * optimized_weights['z'] +z'] * optimized_weights['z'] + 
              metrics['cusum'] * optimized_ 
              metrics['cusum'] * optimized_weights['cusum'] + 
              metricsweights['cusum'] + 
              metrics['bayes']['bayes'] * optimized * optimized_weights['bay_weights['bayes'] +
es'] +
              metrics['              metrics['momentum'] *momentum'] * optimized_weights['momentum'] + optimized_weights['momentum'] + 
              metrics['eor'] * optimized 
              metrics['eor']_weights['eor'])

 * optimized_weights['eor'])

    # 投票    # 投票兜底兜底
    m = metrics
    m = metrics
   
    def sgn(x): def sgn(x): return ' return 'B' ifB' if x>0 else ('P' if x<0 else 'HOLD x>0 else ('P' if x<0 else 'HOLD')
    votes = [sgn')
    votes = [sgn(m['z(m['z']), sgn']), sgn(m['cus(m['cusum']),um']), sgn(m['momentum']), sgn sgn(m['momentum']), sgn(m['bayes(m['bayes']),']), sgn(m['eor'])]
    cnt = sgn(m['eor'])]
    cnt = Counter([ Counter([v for v inv for v in votes if v votes if v!='HOLD'])
    vote!='HOLD'])
    vote_dir,_dir, vote_num = ( vote_num = (None,0None,0) if not) if not cnt else cnt.m cnt else cnt.most_common(1)[0]

    #ost_common(1)[0]

    # 初判
 初判
    if    if hybrid > threshold: prelim hybrid > threshold: prelim = "B"
    elif hybrid < -threshold: prelim = "P = "B"
    elif hybrid < -threshold: prelim ="
    else: prelim = "P"
    else: "HOLD"

    # prelim = "HOLD"

 HOLD 限频策略
    # HOLD 限频    hist = st.session_state.p策略
    hist = st.session_state.prediction_stats.getrediction_stats.get('pred('prediction_history',iction_history', [])
    [])
    recent_window = hist recent_window = hist[-40:] if len(hist)>=[-40:] if len(hist)40 else hist
    hold_>=40 else hist
    hold_ratio_recent =ratio_recent = np.mean([1 if h['pred np.mean([1 if h['prediction']=='HOLDiction']=='HOLD' else 0 for h in' else 0 for h in recent_window]) if recent_window else  recent_window]) if recent_window else 0.0
0.0
    hold    hold_cap =_cap = st.session_state st.session_state.hold_cap_.hold_cap_ratio

    direction = prelim
   ratio

    direction = prelim
    base_conf = 0 base_conf = 0.52.52 + 0 + 0.36*(1/(.36*(1/(1 +1 + np.exp(-abs(hy np.exp(-abs(hybrid)/0.12)))

    ifbrid)/0.12)))

    if hold_ratio_recent hold_ratio_recent > > hold_cap:
        threshold hold_cap:
        threshold *= *= 0.90
        0.90
        if if direction == "HOLD" direction == "HOLD" and vote_dir in ['B and vote_dir in ['B','P'] and vote_num','P'] and vote_num >= >= 3:
            direction = 3:
            direction = vote vote_dir
            base_conf =_dir
            base_conf = max(base_conf, 0 max(base_conf, 0.56)

    #.56)

    #  边际反转
    margin = abs(hy边际反转
    margin = abs(hybrid) -brid) - threshold
    if prelim != "HOLD" and margin <  threshold
    if prelim != "HOLD" and margin < 0.04 and vote_dir0.04 and vote_dir in ['B','P'] and in ['B','P'] and vote vote_dir != prelim:
        direction_dir != prelim:
        direction = vote_dir

 = vote_dir

    # 多时间框架确认 & 模式    # 多时间框架确认 & 强度增强
    direction, base模式强度增强
    direction,_conf = EnhancedLogicCore.m base_conf = EnhancedLogicCoreulti_timeframe_confirmation(seq.multi_timeframe_confirmation, direction, base_conf)
(seq, direction, base    patterns = AdvancedPatternDetector.detect_all_patterns(seq)
    pattern_strength_conf)
    patterns = AdvancedPatternDetector.detect_all_patterns(seq)
    pattern_strength = EnhancedLogicCore. = EnhancedLogicCore.quantify_pattern_strength(patternsquantify_pattern_strength(, st.session_state.expert_patterns, st.session_state.exroads)
    if direction !=pert_roads)
    if 'HOLD':
        base direction != 'HOLD':
        base_conf = min(0._conf = min(0.95, base_conf * (95, base_conf * (1.0 + pattern_stre1.0 + pattern_stngth))

    # 状态信号rength))

    # 状态增强
    if state_sign信号增强
    if state_signals:
        forals:
        for sig in state_signals:
            sig in state_signals:
            if '突破' in sig or '共振 if '突破' in sig or '共振' in sig:
' in sig:
                base                base_conf = min(0.94, base_conf*1_conf = min(0.94, base_conf*1.12)
            if '衰竭.12)
            if '' in sig and direction != '衰竭' in sig and directionHOLD':
                != 'HOLD':
                direction='HOLD'
                base_conf direction='HOLD'
                base_conf=max(base_conf,0.60)

    # 展示参数
    col1, col=max(base_conf,0.60)

    # 展示参数
    col1, col2,2, col3, col col3, col4 = st.columns(44 = st.columns(4)
)
    with col1: st    with col1: st.metric("动态阈值",.metric("动态阈值", f"{ f"{threshold:.3fthreshold:.3f}")
   }")
    with col2: st with col2: st.m.metric("熵值", fetric("熵值", f"{st.session_state.ai_"{st.session_state.ai_entropy:.3entropy:.3f}")
    with col3: st.metricf}")
    with col3: st("HOLD近40.metric("HOLD近占比", f"{hold_ratio_re40占比", f"{hold_ratio_recent*100cent*100:.1:.1f}%f}%")
   ")
    with col4: st.metric with col4: st.metric("投票("投票多数", f"{(多数", f"{(votevote_dir or '—_dir or '—')}({vote')}({vote_num}/5)")
_num}/5)")
    st    st.markdown('</.markdown('</div>',div>', unsafe_allow unsafe_allow_html=True)

   _html=True)

    #  # 看路推荐
看路推荐
    road_s    road_sug = roadug = road_recommendation_recommendation(st.session(st.session_state.expert__state.expert_roads)
roads)
    if road_s    if road_sug and road_sug.get("ug and road_sug.get("final"):
       final"):
        st.markdown st.markdown(f"""
        <div style="background:linear-gradient(90deg,#(f"""
        <div style="backgroundFFD70033,#FF634:linear-gradient(90deg,#FFD70033,#FF733);padding:10px 634733);padding:10px 14px14px;border-radius:;border-radius:10px10px;margin-top:;margin-top:6px;margin-bottom:10px;border-left:5px solid #FFD700;color:#6px;margin-bottom:10px;border-left:5px solid #FFD700;color:#fff;font-weightfff;font-weight:600:600;text-shadow:1px;text-shadow:1px 1px 2px 1px 2px #000;">
            🛣 #000;">
            🛣️ 看路推荐️ 看路推荐：{：{road_sug['finalroad_sug['final']']}
        </div>
       }
        </div>
        """, unsafe_allow_html """, unsafe_allow_html=True)

    # 状态信号展示
    if state=True)

    # 状态信号展示
    if state_signals_signals:
        for s:
        for s in state_signals:
            st.mark in state_signals:
           down(f'< st.markdown(f'div class="state-signal"><div class="state-signal">🚀 状态信号：{s}</div>', unsafe_allow_html=True)

    # 预测卡片
    if🚀 状态信号：{s}</div>', unsafe_allow_html=True)

    # 预测卡片
    if direction=="B":
        color="#FF6 direction=="B":
        colorB6B"; icon="="#FF6B6B"; icon="🔴";🔴"; text="庄(B)"; bg="linear-gradient( text="庄(B)"; bg="linear-gradient(135deg,#135deg,#FF6BFF6B6B,#C445696B,#C44569)"
    elif direction)"
    elif direction=="P":
=="P":
        color        color="#4ECDC4"; icon="="#4ECDC4"; icon🔵";="🔵 text="闲(P"; text="闲(P))"; bg="linear"; bg="linear-gradient(135deg,#4ECDC4,#-gradient(135deg,#4ECDC4,#44A08D)"
    else44A08D)"
    else:
        color="#FFE66D";:
        color="#FFE66D"; icon=" icon="⚪"; text="⚪"; text="观望";观望"; bg="linear-gradient bg="linear-gradient(135deg(135deg,#FFE,#FFE66D,#F66D,#F9A8269A826)"

    # 风险显示
    vol)"

    # 风险显示
    vol = float(abs( = float(abs(metrics['metrics['momentum']))*momentum']))*0.60.6 + 0 + 0.4*(1.4*(1 - abs(metrics - abs(metrics['bay['bayes']es']))
    risk_level, risk_text = ProfessionalRisk))
    risk_level, risk_text = ProfessionalRiskManager.get_risk_level(base_conf, vol)

    st.markdown(f"""
    <div classManager.get_risk_level(base_conf, vol)

    st.markdown(f"""
   ="prediction-card" style <div class="prediction-card"="background:{bg}; style="background:{bg">
        <h2 style="};">
        <h2 stylecolor:{color};margin:="color:{color};margin:0;text-align:center0;text-align:center;">{icon} 大师推荐;">{icon} 大师: {text}</h2>
推荐: {text}</h        <h3 style="2>
        <h3color:#fff;text-align style="color:#fff;:center;margin:10text-align:center;margin:px 0;">🎯10px 0;">🎯 置信度: 置信度: {base_conf*100:.1f}% {base_conf*100:.1f | {risk_text}</h}% | {risk_text3>
        <p style}</h3>
       ="color:#f8f <p style="color:#f89fa;text-align:centerf9fa;text-align:;margin:0;">
            模式center;margin:0;">
            模式: {",".join(: {",".join(patterns[:3]) if patternspatterns[:3]) if patterns else " else "—"} | 风险: {risk_level}
        </p>
    </div>
    """, unsafe_allow—"} | 风险: {risk_level}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # _html=True)

    # 指标指标表
    st.mark表
    st.markdown("#### 📐 Hybrid 指标down("#### 📐 Hybrid总览")
    def badge 指标总览")
   (v):
        v=float def badge(v):
        v(v)
        if v>=float(v)
        if v0>0: return f'<span class="badge badge: return f'<span class="badge badge--pos">+{vpos">+{v:.3f}</span>:.3f}</span'
        if v<0>'
        if v<0: return f'<span class=": return f'<span class="badge badge-neg">{v:.3badge badge-neg">{v:.3f}</span>f}</span>'
       '
        return f' return f'<span class<span class="badge badge-neutral="badge badge-neutral">{">{v:.3f}</spanv:.3f}</span>>'
    w = st'
    w = st.session_state.session_state.ai_.ai_weights
    tbl = f"""
    <divweights
    tbl = f"""
    class="metric-table">
      < <div class="metric-table">
      <div class="rowdiv class="row"><div"><div>Z-S>Z-Score</divcore</div><div>{badge><div>{badge(metrics(metrics['z'])}['z'])} · w={w['z']:.2 · w={w['z']:.2ff}</div></div>
     }</div></div>
      <div <div class="row class="row"><div>CUSUM"><div>CUSUM</div</div><div>{bad><div>{badge(ge(metrics['cusum'])} · w={w['metrics['cusum'])} · w={w['cusumcusum']:.2f}</']:.2f}</div></div></div>
     div>
      <div class="row"><div <div class="row"><div>Bayes</div><div>{badge(metrics['bayes>Bayes</div><div>{badge(metrics['bayes'])} ·'])} · w={w[' w={w['bayesbayes']:.2f']:.2f}</div></div>
      <div class}</div></div>
      <div class="row="row"><div>Moment"><div>Momentum</divum</div><div>{><div>{badge(metricsbadge(metrics['moment['momentum'])} · w={um'])} · w={w['momentum']:.w['momentum']:.2f}</div></div>
     2f}</div></div>
      <div class="row <div class="row"><"><div>EOR+ (div>EOR+ (decks={st.session_state.edecks={st.session_state.eor_decksor_decks})</div><div>{badge(metrics['eor'])} · w})</div><div>{badge(metrics['eor'])} · w={w['eor']:.={w['eor']:.2f}</div></div2f}</div></div>
      <div class="row>
      <div class="row"><div>Entropy</div"><div>Entropy</><div>{badge(st.sessiondiv><div>{badge(st_state.ai_entropy)}</.session_state.ai_entropy)}div></div>
     </div></div>
      <div class=" <div class="row"><div><b>row"><div><b>Hybrid 合成</b></divHybrid 合成</b></div><div><b>{badge(><div><b>{badge(hybrid)}</b></hybrid)}</b></divdiv></div>
      <div></div>
      <div class="row"><div> class="row"><div>方向</div><div方向</div><div><b>{'庄(B)' if><b>{'庄(B)' direction=='B' else (' if direction=='B' else ('闲(P)' if direction闲(P)' if direction=='=='P' else '观望')}</P' else '观望')}</b></div></div>
b></div></div>
    </div>
    """
    </div>
    """
       st.markdown(tbl, st.markdown(tbl, unsafe unsafe_allow_html=True)

   _allow_html=True)

    # 风险控制
    # 风险控制
    st.markdown("### st.markdown("### 🛡️  🛡️ 风险控制")
风险控制")
    pos =    pos = EnhancedLogicCore. EnhancedLogicCore.risk_risk_aware_position_sizing(baseaware_position_sizing(base_conf, direction, metrics_conf, direction, metrics, st.session_state, st.session_state.risk.risk_data['win_stre_data['win_streakak'])
    sug = ProfessionalRisk'])
    sug = ProfessionalRiskManager.getManager.get_trading_suggestion_trading_suggestion(risk_level, direction)
    st(risk_level, direction)
.markdown(f"""
       st.markdown(f"""
 <    <div class="risk-panel">
        <h4 stylediv class="risk-panel">
        <h4 style="color:#fff;margin:0="color:#fff;margin:0 0 10 0 10px px 0;">📊0;">📊 风险 风险控制建议</h控制建议</h4>
        <p style="color:#ccc4>
        <p style="color:#ccc;margin:5px ;margin:5px 0;"><strong>仓位建议0;"><strong>仓位建议:</strong> {pos:.:</strong> {pos:.1f1f} 倍基础仓位</p>
        <p style="color:#} 倍基础仓位</p>
        <p style="colorccc;margin:5px 0:#ccc;margin:5px 0;"><strong>操作建议:</strong;"><strong>操作建议:</strong> {> {sug}</p>
        <p style="color:#ccc;sug}</p>
        <p style="color:#ccc;margin:5px 0;"><strongmargin:5px 0;"><strong>连>连赢:</strong> {st赢:</strong> {st.session_state.risk_data.session_state.risk_data['['win_streak']}win_streak']} 局 局 | <strong> | <strong>连输连输:</strong>:</strong> { {st.session_state.risk_data['consecutivest.session_state.risk_data_losses']} 局</p>
    </div['consecutive_losses']} 局</p>
   >
    """, unsafe_ </div>
    """, unsafe_allow_html=True)

   allow_html=True)

    # # 在线学习
    if 在线学习
    if len(seq) >  len(seq) > 0 and direction != 'HOLD':
0 and direction != 'HOLD        last_result = seq[-':
        last_result = seq[-1]
        record_prediction1]
        record_prediction_result_result(direction, last_result, base_conf)
        enhanced(direction, last_result, base_conf)
        enhanced__learning_update(direction, last_resultlearning_update(direction, last_result)
        st.session_state.last)
        st.session_state.last_prediction = direction

#_prediction = direction

# ========================== 六路 ========================== 六路展示 / 统计 / 历史（展示 / 统计 / 保留） ==========================
def display_com历史（保留） ==========================
plete_roads():
    roadsdef display_complete_roads():
    roads=st.session_state=st.session_state.expert_.expert_roads
    st.markdown("##roads
    st.markdown("## 🛣️  🛣️ 完整六路分析")
    st.markdown("####完整六路分析")
    st.markdown("#### 🟠 珠 🟠 珠路 (最近20路 (最近20局)")
局)")
    if roads['    if roads['bead_bead_road']:
road']:
        disp=" ".        disp=" ".join(["🔴join(["🔴" if" if x=='B' x=='B' else " else "🔵" for🔵" for x in x in roads['bead roads['bead_road'][_road'][-20:]-20:]])
       ])
        st.markdown(f' st.markdown(f'<div class<div class="road="road-display">{disp}</div>',-display">{disp}</div>', unsafe_ unsafe_allow_html=True)
allow_html=True)
    st.mark    st.markdown("#### 🔴 大路")
    ifdown("#### 🔴 大路")
    if roads[' roads['big_road']big_road']:
       :
        for i,col in for i,col in enumerate( enumerate(roads['big_road'][-6:]):
           roads['big_road'][-6:]):
            col_disp=" ".join(["🔴" if col_disp=" ".join(["🔴" if x=='B' else "🔵" for x in col])
            st.markdown x=='B' else "🔵" for x in col])
            st.markdown(f'<(f'<div classdiv class="road-display">第{i+="road-display">第{i+11}列: {col_d}列: {col_disp}</div>isp}</div>', unsafe_allow_html', unsafe_allow_html=True)
    c1,c2==True)
    c1,c2=st.columnsst.columns(2)
    with c1(2)
    with c1:
        if roads['big:
        if roads['big_eye_road']:
_eye_road']:
            st.markdown("#### 👁            st.markdown("#### 👁️ 大眼路")
️ 大眼路")
            disp=" ".join(["            disp=" ".join(["🔴" if x=='R' else "🔵"🔴" if x=='R' else "🔵" for x in roads['big_ for x in roads['big_eye_road'][-12eye_road'][-12:]])
            st.markdown(f':]])
            st.markdown(f'<div class="<div class="road-disroad-display">{play">{disp}</div>', unsafe_allow_html=True)
    with cdisp}</div>', unsafe_allow_html=True)
    with c2:
        if2:
        if roads['small_ roads['small_road']:
            st.markroad']:
            st.markdown("#### 🔵 down("#### 🔵 小路")
小路")
            disp=" ".join            disp=" ".join(["🔴" if x=='(["🔴" if x=='R' else "🔵"R' else "🔵 for x in roads['small_" for x in roads['road'][-10:]])
small_road'][-10            st.markdown(f':]])
            st.markdown<div class="road-dis(f'<div class="road-display">{disp}</play">{disp}</div>',div>', unsafe_allow_html=True unsafe_allow_html=True)
   )
    if roads['three if roads['three_bead_bead_road']:
_road']:
        st        st.markdown("#### 🔶.markdown("#### 🔶 三珠路")
        for i 三珠路")
        for i,g in enumerate(,g in enumerate(roads['three_bead_roadroads['three_bead_road'][-6:]'][-6:]):
            disp="):
            disp=" ".join(["🔴 ".join(["🔴"" if x==' if x=='B' else "🔵" forB' else "🔵" x in g])
            st.markdown for x in g])
            st(f'<div class=".markdown(f'<divroad-display">第{i+ class="road-display">第{i+1}组1}组: {: {disp}</div>', unsafedisp}</div>', unsafe__allow_html=True)

defallow_html=True)

def display_professional_stats():
    if not st.session display_professional_stats():
    if not st.session_state.ultimate_games:
        st.info("暂无统计数据"); return
    games_state.ultimate_games:
        st.info("暂无统计数据"); return
    games=st.session_state.ultimate=st.session_state.ultimate_games_games; results=[g['; results=[g['result'] for g inresult'] for g in games]
    bead games]
    bead=st.session_state.expert_roads['bead=st.session_state.expert_roads['be_road']
   ad_road']
 st.markdown("## 📊 专业    st.markdown("## 📊 专业统计")
    c1,c2,c统计")
    c1,c2,c3,c4=3,c4=st.columnsst.columns(4)
    with c(4)
    with c1: st.metric("1: st.metric("总局数", len(results))
总局数", len(results))
    with    with c2: st c2: st.m.metric("庄赢", resultsetric("庄赢", results.count.count('B'))
    with c('B'))
    with c3: st.metric("3: st.metric("闲赢", results.count('闲赢", results.count('P'))
    with c4: st.metric("和局P'))
    with c4: st.metric("和局",", results.count('T'))
    if bead:
        st.mark results.count('T'))
    if bead:
       down("#### 📈  st.markdown("#### 📈 高级分析")
        d1,d高级分析")
        d1,d2,d3=st2,d3=st.columns(3)
        with d1.columns(3)
        with d1:
            total=len(results:
            total=len(results)
            if total>0)
            if total>0: st.metric("庄: st.metric("庄胜率", f"{results.count('B胜率", f"{results.count('B')/total*100:.1f}%")
')/total*100:.1f}%")
        with d2:
            avg        with d2:
           =np.mean([len(list(g avg=np.mean([len(list(g)) for k,g in groupby)) for k,g in groupby(bead)]) if(bead)]) if len(bead)>0 else 0
            st.m len(bead)>0 else 0
            st.metric("平均连赢", fetric("平均连赢", f"{avg:.1f}"{avg:.1f}局")
        with d3局")
        with d3:
            if len(be:
            if len(bead)>1:
                changes=sum(1 for i inad)>1:
                changes=sum(1 for i in range(1,len(bead range(1,len(bead)) if bead[i]!=be)) if bead[i]!=bead[i-1])
               ad[i-1])
                vol=changes/len(be vol=changes/len(bead)*100
                st.mad)*100
                stetric("波动率", f.metric("波动率", f"{vol:.1f}"{vol:.1%")
    stats = stf}%")
    stats = st.session_state.session_state.prediction_stats
    if stats['total_predict.prediction_stats
    if stats['totalions'] > _predictions'] > 0:
0:
        st        st.markdown(".markdown("#### 🎯 AI预测性能#### 🎯 AI预测性能")
        col1, col")
        col1, col2,2, col3 = st col3 = st.columns(3)
.columns(3)
        with        with col1:
            col1:
            accuracy = accuracy = (stats['correct (stats['correct_predictions'] / stats['total_p_predictions'] / stats['total_predictions'])redictions']) * 100
 * 100
            st            st.metric("总体.metric("总体准确率",准确率", f"{accuracy:.1f}%")
 f"{accuracy:.1f}%")
        with col2:
                   with col2:
            recent recent_acc = np.mean(stats['recent_accuracy'][-20:]) * 100_acc = np.mean(stats['recent_accuracy'][-20:]) * 100 if stats['re if stats['recent_accuracy']cent_accuracy'] else 0
            st.metric(" else 0
            st.metric("近期准确率", f"{recent_acc:.1f}%近期准确率", f"{recent_acc:.1f")
        with col3:
            st.metric("总预测}%")
        with col3:
            st.metric("总预测数", stats['total_predict数", stats['total_predictions'])
        st.markions'])
        st.markdown("down("#### 🤖 指标#### 🤖 指标性能分析")
        perf_cols性能分析")
        perf_cols = st.columns(5 = st.columns()
        for i, metric in5)
        for i, metric in enumerate(['z','cusum enumerate(['z','cusum','bay','bayes','momentumes','momentum','','eor']):
            with perf_cols[i]:
                if steor']):
            with perf_cols[i]:
                if st.session.session_state.weight_performance_state.weight_performance[metric]:
                    perf = np.mean[metric]:
                    perf = np.mean(st(st.session_state.weight_performance.session_state.weight_performance[metric][-10:])[metric][-10:]) * 100
                    st.metric(f"{metric. * 100
                    st.metric(f"{metric.upper()}upper()}准确率", f"{per准确率", f"{perf:.1ff:.1f}}%")

def display_complete_history%")

def display_complete_history():
    if not st.session():
    if not st.session_state.ultimate_games:
        st_state.ultimate_games:
        st.info("暂无历史记录");.info("暂无历史记录"); return
    st.markdown return
    st.markdown("## 📝 完整历史")
    recent=st("## 📝 完整历史")
    recent=st.session_state..session_state.ultimate_games[-10:]
    for g in reversedultimate_games[-10:]
    for g in reversed(recent(recent):
        icon="):
        icon="🃏🃏" if" if g.get('mode') g.get('mode')=='=='card' elsecard' else ("🎯" if ("🎯" if g.get('mode')=='quick' else " g.get('mode')=='quick' else "📝")
📝")
               with st.container():
            c1,c2,c3,c with st.container():
            c1,c2,c3,c4,c5=st4,c5=st.columns.columns([1,1,([1,1,2,2,2,1])
2,1])
            with c            with c1: st.write1: st.write(f"#{g['round'](f"#{g['round}")
            with c2: st.write']}")
            with c2:(icon)
            with c st.write(icon)
            with c3: st.write3: st.write(f"闲(f"闲: {'-'.: {'-'.join(gjoin(g['player_cards['player_cards'])}'])}" if g.get('mode')=='card' else "快速记录")
            with c4: st.write(f"" if g.get('mode')=='card' else "快速记录")
            with c4: st.write(f"庄: {'-'.庄: {'-'.joinjoin(g['bank(g['banker_cardser_cards'])}" if g'])}" if g.get('mode')=='card' else ".get('mode')=='card' else "快速记录")
            with快速记录")
            with c5:
 c5:
                if g['                if g['result']=='B': st.error("result']=='B': st庄赢")
                elif g.error("庄赢")
                elif g['result']=='['result']=='P': stP': st.info("闲.info("闲赢")
赢")
                else: st.w                else: st.warningarning("和局")

#("和局")

# ========================== ========================== 界面（优化版） ==========================
def display_complete_interface():
 界面（优化版） ==========================
def display_complete_interface():
    st.markdown    st.markdown("##("## 🎮 双模式 🎮 双模式输入输入系统")
    show_系统")
    show_quick_start_guide()
    
    cquick_start_guide()
    
1, c2 = st.columns    c1, c2 = st.columns(2)
   (2)
    with c with c1:
        if1:
        if st.button(" st.button("🃏 牌点输入", use_container_width🃏 牌点输入", use_container_width=True=True, type, type="primary"):
            st="primary"):
            st.session_state.input_mode.session_state.input_mode='card'; st.rerun='card'; st.rerun()
    with c2:
        if()
    with c2:
        if st.button("🎯 快速看 st.button("🎯 快速路", use_container_width=True):
看路", use_container_width=True            st):
            st.session_state.input.session_state.input_mode='result'; st.r_mode='result'; st.rerun()
            
    if "inputerun()
            
    if "input_mode" not in st_mode" not in st.session_state: 
        st.session.session_state: 
        st.session_state.input_mode='card'
_state.input_mode='        
    if st.session_state.input_modecard'
        
    if st.session_state.input_mode == 'card':
        == 'card':
        st.markdown st.markdown("###("### 🃏 🃏 详细牌点记录 详细牌点记录")
        
")
        
        # 使用        # 使用新的扑克牌按钮界面
        player新的扑克牌按钮界面_input, banker_input =
        player_input, banker_input = card_button_interface card_button_interface()
        
        st.markdown("###()
        
        st.markdown("### 🏆 本局结果")
 🏆 本局结果")
        b1,        b1, b2 b2, b3 =, b3 = st.columns(3)
        with b1: st.columns(3)
        with b1: 
            banker_btn 
            banker_btn = st.button("🔴  = st.button("🔴 庄赢庄赢", use_container_width=True", use_container_width=True, type="primary")
        with b, type="primary")
2: 
            player_btn        with b2: 
            player = st.button("🔵_btn = st.button(" 闲赢", use_container🔵 闲赢", use_container_width=True)
        with b_width=True)
        with b3: 
            tie_3: 
            tie_btn = st.button("⚪ btn = st.button("⚪和局", use_container_width 和局", use_container_width=True)
            
        if banker=True)
            
        if banker_btn or player_btn or tie_btn:
           _btn or player_btn or tie_btn:
            handle_card handle_card_input(player_input, banker_input, banker_btn_input(player_input, banker_input, banker_btn, player_btn, tie_btn, player_btn, tie_)
            
    else:
       btn)
            
    else:
        st.markdown("### st.markdown("### 🎯 快速结果记录 🎯 快速结果记录")
")
        st.info("        st.info("💡 💡 快速模式：直接记录结果快速模式：直接记录结果，用于快速看路分析，用于快速看路分析")
        
        q1,")
        
        q1, q2, q3 = st.columns q2, q3 = st.columns(3)
        with q(3)
        with q1: 
            qb1: 
            qb = st.button("🔴 = st.button("🔴 庄赢", use_container_width=True 庄赢", use_container_width=True, type="primary")
       , type="primary")
        with with q2: 
            qp = st.button(" q2: 
            qp = st.button("🔵 闲赢",🔵 闲赢", use_container_width use_container_width=True)
=True)
        with q3: 
                   with q3: 
            qt = st.button(" qt = st.button("⚪ 和局", use_container_width=True)
            
        if qb:⚪ 和局", use_container_width=True)
            
        
            handle_quick_input(True, False)
        if qb: 
            handle_quick_input(True, False)
        if qp: if qp: 
            
            handle_quick_input handle_quick_input(False(False, True)
        if, True)
        if qt: 
            record_game('T qt: 
            record_game('T', ['X','X', ['X','X'],'], ['X','X'], ['X','X'], 'quick')

# ========================== 主 'quick')

# =========================程序 ==========================
def main= 主程序 ==========================
():
    with st.sidebardef main():
    with st.sidebar:
        st.mark:
        st.markdown("down("## ⚙## ⚙️ ️ 控制台")
        st.caption("动态优化AI权重控制台")
        st.caption("动态优化AI权重，自适应市场环境，自适应市场环境；E；EOR+ 已启用OR+ 已启用；；HOLD≤15% HOLD≤15% 限频。")
        add_system限频。")
        add_status_panel()

    tab_system_status_panel()

    tab1, tab21, tab2, tab3, tab4 = st.t, tab3, tab4 = st.tabs(["🎯 abs(["🎯 智能分析智能分析", "🛣️ 六", "🛣️ 六路分析", "📊 专业路分析", "📊 专业统计",统计", "📝  "📝 历史记录历史记录"])
    with tab"])
    with tab1:
        display_complete_interface1:
        display_complete_interface()
        st.markdown("()
        st.markdown("---")
        display_complete---")
        display_complete_analysis_analysis()
    with tab()
    with tab2:
2:
        display_complete_roads()
    with tab3:
        display_complete_roads()
    with tab3:
        display        display_professional_stats_professional_stats()
   ()
    with tab4:
        display_complete_history()

    with tab4:
        display_complete_history()

    st st.markdown("---")
   .markdown("---")
    c1 c1, c, c2 = st.columns2 = st.columns(2)
   (2)
    with c1:
        if with c1:
        if st.button("🔄 开始新 st.button("🔄 开始牌靴", use_container_width新牌靴", use_container_width=True):
            st.session_state.ultimate_games.clear()
            st=True):
            st.session_state.ultimate_games.clear()
            st.session_state.expert.session_state.expert_roads_roads = {
                'big_ = {
                'big_road': [],
                'bead_road': [],
                'beadroad': [],
                '_road': [],
                'big_eye_big_eye_road': [],
road': [],
                               'small_road': [],
                ' 'small_road': [],
                'cockroach_road': [],
                'three_becockroach_road': [],
                'three_bead_road': []
            }
           ad_road': []
            }
 st.session_state.risk_data = {
                'current            st.session_state.risk_data = {
                '_level': 'medium',
                'current_level': 'medium',
                'position_size': 1.position_size': 1.0,
                'stop_loss0,
                'stop_loss': 3,
                'consecutive_losses': 0': 3,
                'consecutive_losses': 0,
                'win_stre,
                'win_streak': 0
           ak': 0
            }
            st.session_state.player }
            st.session_state.player_c_cards_input = ""
ards_input = ""
            st.session_state.banker_cards_input = ""
            st.success("新牌靴开始！"); st            st.session_state.banker_cards_input = ""
.rerun()
    with            st.success("新牌靴开始！"); st.rerun()
    with c2:
        if st.button(" c2:
        if st.button📋 导出数据", use("📋 导出数据", use_container_width=True):
_container_width=True):
            enhanced            enhanced_export_data()

if_export_data()

if __name__ == "__main__":
    main __name__ == "__main__":
()
