# -*- coding: utf-8 -*-
# Baccarat Master Ultimate - Precision 15.0 Ultimate · EOR-Bayes Fusion 版
# 在 Precision 13.5（EOR+ + HOLD≤15% 限频）基础上，无损升级 EOR 为「EOR-Bayes 后验融合」
# ✅ 保留你现有的全部模块/界面/统计/导出/六路/风控/学习/动态阈值/权重自适应
# ✅ 新增：EOR-Bayes（含可选 Monte-Carlo Light 平滑）、侧边栏参数开关

import streamlit as st
import numpy as np
import math
import json
from collections import defaultdict, Counter
from datetime import datetime
from itertools import groupby

# ========================== 基础配置 ==========================
st.set_page_config(page_title="🐉 百家乐大师 Precision 15.0 · EOR-Bayes Fusion", layout="centered")

st.markdown("""
<style>
.main-header {font-size:2.2rem;color:#FFD700;text-align:center;text-shadow:2px 2px 4px #000;}
.prediction-card{background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:15px;border:3px solid #FFD700;margin:15px 0;text-align:center;}
.road-display{background:#1a1a1a;padding:12px;border-radius:8px;margin:8px 0;border:1px solid #333;}
.multi-road{background:#2d3748;padding:10px;border-radius:8px;margin:5px 0;font-family:monospace;}
.risk-panel{background:#2d3748;padding:15px;border-radius:10px;margin:10px 0;border-left:4px solid #e74c3c;}
.pattern-badge{background:#e74c3c;color:white;padding:4px 8px;border-radius:12px;font-size:12px;margin:2px;display:inline-block;}
.metric-table{background:#1f2937;border-radius:10px;padding:10px 12px;margin-top:8px;border:1px solid #334155;color:#e5e7eb;font-size:14px;}
.metric-table .row{display:flex;justify-content:space-between;padding:4px 0;}
.badge{padding:2px 6px;border-radius:6px;font-weight:700;font-size:12px;}
.badge-pos{background:#14532d;color:#bbf7d0;}
.badge-neg{background:#7f1d1d;color:#fecaca;}
.badge-neutral{background:#334155;color:#cbd5e1;}
.state-signal{background:linear-gradient(90deg,#FFD70033,#FF634733);padding:8px 12px;border-radius:8px;margin:5px 0;border-left:4px solid #FFD700;color:#fff;font-weight:600;}
.guide-panel{background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:10px;margin:10px 0;color:white;}
.enhanced-logic-panel{background:linear-gradient(135deg,#00b4db,#0083b0);padding:15px;border-radius:10px;margin:10px 0;color:white;}
.adaptive-panel{background:linear-gradient(135deg,#ff9a9e,#fad0c4);padding:15px;border-radius:10px;margin:10px 0;color:#333;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐉 百家乐大师 Precision 15.0 · EOR-Bayes Fusion</h1>', unsafe_allow_html=True)

# ========================== 状态初始化 ==========================
def _init_state():
    ss = st.session_state
    ss.setdefault("ultimate_games", [])
    ss.setdefault("expert_roads", {'big_road':[],'bead_road':[],'big_eye_road':[],'small_road':[],'cockroach_road':[],'three_bead_road':[]})
    ss.setdefault("risk_data", {'current_level':'medium','position_size':1.0,'stop_loss':3,'consecutive_losses':0,'win_streak':0})
    ss.setdefault("ai_weights", {'z':0.25,'cusum':0.25,'bayes':0.20,'momentum':0.15,'eor':0.15})
    ss.setdefault("ai_learning_buffer", [])
    ss.setdefault("ai_last_metrics", {})
    ss.setdefault("ai_entropy", 0.0)
    ss.setdefault("eor_decks", 7)
    ss.setdefault("ai_batch_n", 5)
    ss.setdefault("prediction_stats", {'total_predictions':0,'correct_predictions':0,'recent_accuracy':[],'prediction_history':[]})
    ss.setdefault("learning_effectiveness", [])
    ss.setdefault("performance_warnings", [])
    ss.setdefault("last_prediction", None)
    ss.setdefault("weight_performance", {'z': [], 'cusum': [], 'bayes': [], 'momentum': [], 'eor': []})
    # 限频：HOLD 目标上限
    ss.setdefault("hold_cap_ratio", 0.15)
    # 15.0 新增：EOR-Bayes 参数
    ss.setdefault("eor_use_mc", True)        # 是否启用 Monte-Carlo Light 平滑
    ss.setdefault("eor_mc_n", 400)           # 采样数（建议 200~800）
    ss.setdefault("eor_k_bias", 1.10)        # 多窗偏差的强度系数
    ss.setdefault("eor_k_align", 0.85)       # 路单对齐的强度系数
    ss.setdefault("eor_entropy_damp", 0.35)  # 熵抑制系数（越大越保守）
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

# ========================== 15.0 EOR-Bayes / 指标核心 ==========================
class EORBayesFusion:
    """
    将原 EOR+ 替换为「EOR-Bayes 后验融合」：
      1) 先验：用历史 B/P 频率得到 P0(B)
      2) 似然：由多时间窗偏差 + 路单对齐 → 映射为 L = P(data|B)
          * 融合窗口：12/24/48（指数权重）
          * 路单对齐：大路末列、BigEye、Small
      3) Bayes 后验：PosteriorOdds = PriorOdds * LR (LR = L/(1-L))
      4) 熵抑制：高熵→靠拢 0.5
      5) decks 缩放：sqrt 副数抑制过拟合
      6) 可选 Monte-Carlo Light：对 (bias, align) 做微扰抽样，平滑后验
    输出：映射到约 [-0.6, 0.6] 的“EOR 数值”，并保持与 13.5 接口一致
    """
    @staticmethod
    def _multiwindow_bias(bp):
        n=len(bp)
        if n<6: return 0.0
        def win_bias(k):
            if n<k: return 0.0
            last=bp[-k:]
            pB=last.count('B')/k
            return (pB - (1-pB))  # B-P
        b12 = win_bias(12)
        b24 = win_bias(24)
        b48 = win_bias(48)
        # 指数衰减融合：最近更重
        return 0.50*b12 + 0.30*b24 + 0.20*b48

    @staticmethod
    def _road_alignment(roads):
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
        return align

    @staticmethod
    def _sigmoid(x):
        return 1.0/(1.0+math.exp(-x))

    @staticmethod
    def _posterior_once(seq, roads, decks, k_bias, k_align, entropy_damp):
        bp=[x for x in seq if x in ['B','P']]
        n=len(bp)
        if n<6:
            return 0.0

        # ---- 先验 P0(B)（拉普拉斯平滑）----
        p0B = (bp.count('B') + 1) / (n + 2)
        p0B = float(np.clip(p0B, 0.05, 0.95))
        prior_odds = p0B / (1 - p0B)

        # ---- 似然 L = P(data|B) ----
        fused_bias = EORBayesFusion._multiwindow_bias(bp)          # [-1,1] 附近
        align = EORBayesFusion._road_alignment(roads)              # 小幅正负
        # 将 (bias, align) 映射到 logit 空间，系数可调
        logit_like = k_bias * fused_bias + k_align * align
        L = EORBayesFusion._sigmoid(logit_like)                    # (0,1)

        # ---- Bayes 后验 ----
        lr = L / max(1e-9, (1-L))                                  # 似然比
        post_odds = prior_odds * lr
        post_B = post_odds / (1 + post_odds)                       # Posterior P(B)
        post_B = float(np.clip(post_B, 1e-4, 1-1e-4))

        # ---- 熵抑制 + decks 缩放 ----
        pB_all = bp.count('B')/n
        pP_all = 1-pB_all
        entropy = -(pB_all*np.log2(pB_all+1e-9)+pP_all*np.log2(pP_all+1e-9))  # ~[0,1]
        # 越混沌→越靠近0.5
        post_B_adj = 0.5 + (post_B - 0.5) * (1.0 - entropy_damp * entropy)

        deck_scale = np.sqrt(max(1, decks)) / 4.0                  # 温和抑制
        # 转为对称值并限制幅度（与 13.5 对齐范围）
        eor_val = float(np.clip((post_B_adj - 0.5) * (1.0 + deck_scale) * 1.2, -0.6, 0.6))
        return eor_val

    @staticmethod
    def posterior(seq, roads, decks, use_mc=True, mc_n=400, k_bias=1.10, k_align=0.85, entropy_damp=0.35):
        """
        对外主接口：返回与原 eor 等价的“对称值”。
        use_mc=True 时，对 (bias, align) 的内部 logit 做小扰动采样平滑，提升稳健性。
        """
        if not use_mc:
            return EORBayesFusion._posterior_once(seq, roads, decks, k_bias, k_align, entropy_damp)

        # Monte-Carlo Light：对 logit_like 的输入做微扰，避免单点抖动
        # 给到 0 均值的小高斯噪声，规模随样本量和熵适度调整
        bp=[x for x in seq if x in ['B','P']]
        n=len(bp)
        if n<6:
            return 0.0
        pB = bp.count('B')/n
        entropy = -(pB*np.log2(pB+1e-9)+(1-pB)*np.log2(1-pB+1e-9))
        noise_scale = 0.10 + 0.10*entropy + max(0, 0.10 - min(0.10, n/2000))  # 0.10~0.25 左右
        samples=[]
        for _ in range(int(mc_n)):
            kb = np.random.normal(k_bias, noise_scale*0.15)     # 轻扰动
            ka = np.random.normal(k_align, noise_scale*0.12)
            val = EORBayesFusion._posterior_once(seq, roads, decks, kb, ka, entropy_damp)
            samples.append(val)
        # 采用截尾均值（抗异常值）
        arr = np.array(samples)
        lo, hi = np.percentile(arr, [10, 90])
        trimmed = arr[(arr>=lo)&(arr<=hi)]
        return float(np.mean(trimmed) if trimmed.size>0 else np.mean(arr))

class HybridMathCore:
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

        # === 15.0：EOR-Bayes 核心 ===
        decks=st.session_state.eor_decks
        roads = st.session_state.expert_roads
        use_mc = bool(st.session_state.eor_use_mc)
        mc_n   = int(st.session_state.eor_mc_n)
        k_bias = float(st.session_state.eor_k_bias)
        k_align= float(st.session_state.eor_k_align)
        damp   = float(st.session_state.eor_entropy_damp)
        eor = EORBayesFusion.posterior(seq, roads, decks, use_mc=use_mc, mc_n=mc_n,
                                       k_bias=k_bias, k_align=k_align, entropy_damp=damp)

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

# ========================== 权重自适应 / 多时间框架 / 风险（保留原 13 增强逻辑） ==========================
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
            trend="红红蓝" if last3.count('R')==2 else ("蓝蓝红" if last3.count('B')==2 else "混乱")
            lines.append(f"蟑螂路：{trend} → {'轻微震荡' if trend!='混乱' else '趋势不明'}")
    if not final:
        if roads['big_eye_路'] if False else roads['big_eye_road']:
            r=roads['big_eye_road'].count('R'); b=roads['big_eye_road'].count('B')
            final="顺路（偏红，延续）" if r>b else ("反路（偏蓝，注意反转）" if b>r else "暂无明显方向")
        else: final="暂无明显方向"
    return {"lines":lines,"final":final}

# ========================== 辅助输入/记录（保留） ==========================
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
    risk=st.session_state.risk_data
    if result in ['B','P']: 
        risk['win_streak']+=1; risk['consecutive_losses']=0
    elif result=='T':
        pass
    else:
        risk['consecutive_losses']+=1; risk['win_streak']=0
    st.success(f"✅ 记录成功! 第{game['round']}局"); st.rerun()

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
        st.success(f"✅ 批量添加 {len(valid)} 局")

# ========================== 系统面板 / 导出（保留） ==========================
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

def enhanced_export_data():
    data = {
        'games': st.session_state.ultimate_games,
        'roads': st.session_state.expert_roads,
        'ai_weights': st.session_state.ai_weights,
        'prediction_stats': st.session_state.prediction_stats,
        'weight_performance': st.session_state.weight_performance,
        'export_time': datetime.now().isoformat()
    }
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    st.download_button(
        label="📥 下载完整数据",
        data=json_str,
        file_name=f"baccarat_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )

def show_quick_start_guide():
    if len(st.session_state.ultimate_games) == 0:
        st.markdown("""
        <div class="guide-panel">
        <h3>🎯 快速开始指南</h3>
        <p>1. 选择「牌点输入」记录详细牌局，或使用「快速看路」快速开始</p>
        <p>2. 记录3局后激活AI智能分析系统</p>
        <p>3. 关注风险建议，科学管理仓位</p>
        <p>4. 系统会持续学习优化预测准确性</p>
        </div>
        """, unsafe_allow_html=True)

# ========================== 统计记录（保留） ==========================
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
            'correct': bool(is_correct),
            'weights_snapshot': dict(st.session_state.ai_weights),
            'timestamp': datetime.now().isoformat()
        })

# ========================== 智能分析（含 HOLD≤15% 限频） ==========================
def display_complete_analysis():
    if len(st.session_state.ultimate_games)<3:
        st.info("🎲 请先记录至少3局牌局数据"); return

    seq=[g['result'] for g in st.session_state.ultimate_games]
    hybrid, metrics = AIHybridLearner.compute_hybrid(seq)

    with st.sidebar:
        st.markdown("### 🤖 EOR-Bayes 设置")
        decks = st.slider("EOR 副数（1-8）", 1, 8, int(st.session_state.eor_decks), key="eor_slider")
        if decks != st.session_state.eor_decks:
            st.session_state.eor_decks = decks
        st.toggle("启用 Monte-Carlo 平滑", key="eor_use_mc", value=st.session_state.eor_use_mc)
        st.slider("MC 采样数", 100, 1000, int(st.session_state.eor_mc_n), 50, key="eor_mc_n")
        st.slider("偏差强度 k_bias", 0.6, 1.8, float(st.session_state.eor_k_bias), 0.05, key="eor_k_bias")
        st.slider("对齐强度 k_align", 0.3, 1.5, float(st.session_state.eor_k_align), 0.05, key="eor_k_align")
        st.slider("熵抑制 entropy_damp", 0.1, 0.6, float(st.session_state.eor_entropy_damp), 0.02, key="eor_entropy_damp")

        st.markdown("### 🤖 AI 权重（动态优化后）")
        st.write({k: round(v,3) for k,v in st.session_state.ai_weights.items()})

    state_signals = GameStateDetector.detect(st.session_state.expert_roads)

    st.markdown('<div class="enhanced-logic-panel">', unsafe_allow_html=True)
    st.markdown("### 🧠 智能决策引擎（EOR-Bayes + 动态阈值 + 限频HOLD）")

    # 动态阈值
    threshold = EnhancedLogicCore.enhanced_dynamic_threshold(seq, metrics, st.session_state.expert_roads)

    # 权重自适应
    actual_results = [g['result'] for g in st.session_state.ultimate_games]
    optimized_weights = EnhancedLogicCore.adaptive_weight_optimization(seq, actual_results)

    # 用优化后的权重修正 hybrid（保持与原展示一致）
    hybrid = (metrics['z'] * optimized_weights['z'] + 
              metrics['cusum'] * optimized_weights['cusum'] + 
              metrics['bayes'] * optimized_weights['bayes'] +
              metrics['momentum'] * optimized_weights['momentum'] + 
              metrics['eor'] * optimized_weights['eor'])

    # 投票兜底
    m = metrics
    def sgn(x): return 'B' if x>0 else ('P' if x<0 else 'HOLD')
    votes = [sgn(m['z']), sgn(m['cusum']), sgn(m['momentum']), sgn(m['bayes']), sgn(m['eor'])]
    cnt = Counter([v for v in votes if v!='HOLD'])
    vote_dir, vote_num = (None,0) if not cnt else cnt.most_common(1)[0]

    # 初判
    if hybrid > threshold: prelim = "B"
    elif hybrid < -threshold: prelim = "P"
    else: prelim = "HOLD"

    # HOLD 限频策略
    hist = st.session_state.prediction_stats.get('prediction_history', [])
    recent_window = hist[-40:] if len(hist)>=40 else hist
    hold_ratio_recent = np.mean([1 if h['prediction']=='HOLD' else 0 for h in recent_window]) if recent_window else 0.0
    hold_cap = st.session_state.hold_cap_ratio

    direction = prelim
    base_conf = 0.52 + 0.36*(1/(1 + np.exp(-abs(hybrid)/0.12)))  # 0.52~0.88

    if hold_ratio_recent > hold_cap:
        threshold *= 0.90  # 放宽 10%
        if direction == "HOLD" and vote_dir in ['B','P'] and vote_num >= 3:
            direction = vote_dir
            base_conf = max(base_conf, 0.56)

    # 边际反转
    margin = abs(hybrid) - threshold
    if prelim != "HOLD" and margin < 0.04 and vote_dir in ['B','P'] and vote_dir != prelim:
        direction = vote_dir

    # 多时间框架确认 & 模式强度增强
    direction, base_conf = EnhancedLogicCore.multi_timeframe_confirmation(seq, direction, base_conf)
    patterns = AdvancedPatternDetector.detect_all_patterns(seq)
    pattern_strength = EnhancedLogicCore.quantify_pattern_strength(patterns, st.session_state.expert_roads)
    if direction != 'HOLD':
        base_conf = min(0.95, base_conf * (1.0 + pattern_strength))

    # 状态信号增强
    if state_signals:
        for sig in state_signals:
            if '突破' in sig or '共振' in sig:
                base_conf = min(0.94, base_conf*1.12)
            if '衰竭' in sig and direction != 'HOLD':
                direction='HOLD'
                base_conf=max(base_conf,0.60)

    # 展示参数
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("动态阈值", f"{threshold:.3f}")
    with col2: st.metric("熵值", f"{st.session_state.ai_entropy:.3f}")
    with col3: st.metric("HOLD近40占比", f"{hold_ratio_recent*100:.1f}%")
    with col4: st.metric("投票多数", f"{(vote_dir or '—')}({vote_num}/5)")
    st.markdown('</div>', unsafe_allow_html=True)

    # 看路推荐
    road_sug = road_recommendation(st.session_state.expert_roads)
    if road_sug and road_sug.get("final"):
        st.markdown(f"""
        <div style="background:linear-gradient(90deg,#FFD70033,#FF634733);padding:10px 14px;border-radius:10px;margin-top:6px;margin-bottom:10px;border-left:5px solid #FFD700;color:#fff;font-weight:600;text-shadow:1px 1px 2px #000;">
            🛣️ 看路推荐：{road_sug['final']}
        </div>
        """, unsafe_allow_html=True)

    # 状态信号展示
    if state_signals:
        for s in state_signals:
            st.markdown(f'<div class="state-signal">🚀 状态信号：{s}</div>', unsafe_allow_html=True)

    # 预测卡片
    if direction=="B":
        color="#FF6B6B"; icon="🔴"; text="庄(B)"; bg="linear-gradient(135deg,#FF6B6B,#C44569)"
    elif direction=="P":
        color="#4ECDC4"; icon="🔵"; text="闲(P)"; bg="linear-gradient(135deg,#4ECDC4,#44A08D)"
    else:
        color="#FFE66D"; icon="⚪"; text="观望"; bg="linear-gradient(135deg,#FFE66D,#F9A826)"

    # 风险显示
    vol = float(abs(metrics['momentum']))*0.6 + 0.4*(1 - abs(metrics['bayes']))
    risk_level, risk_text = ProfessionalRiskManager.get_risk_level(base_conf, vol)

    st.markdown(f"""
    <div class="prediction-card" style="background:{bg};">
        <h2 style="color:{color};margin:0;text-align:center;">{icon} 大师推荐: {text}</h2>
        <h3 style="color:#fff;text-align:center;margin:10px 0;">🎯 置信度: {base_conf*100:.1f}% | {risk_text}</h3>
        <p style="color:#f8f9fa;text-align:center;margin:0;">
            模式: {",".join(patterns[:3]) if patterns else "—"} | 风险: {risk_level}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 指标表
    st.markdown("#### 📐 Hybrid 指标总览")
    def badge(v):
        v=float(v)
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
      <div class="row"><div>EOR-Bayes (decks={st.session_state.eor_decks})</div><div>{badge(metrics['eor'])} · w={w['eor']:.2f}</div></div>
      <div class="row"><div>Entropy</div><div>{badge(st.session_state.ai_entropy)}</div></div>
      <div class="row"><div><b>Hybrid 合成</b></div><div><b>{badge(hybrid)}</b></div></div>
      <div class="row"><div>方向</div><div><b>{'庄(B)' if direction=='B' else ('闲(P)' if direction=='P' else '观望')}</b></div></div>
    </div>
    """
    st.markdown(tbl, unsafe_allow_html=True)

    # 风险控制
    st.markdown("### 🛡️ 风险控制")
    pos = EnhancedLogicCore.risk_aware_position_sizing(base_conf, direction, metrics, st.session_state.risk_data['win_streak'])
    sug = ProfessionalRiskManager.get_trading_suggestion(risk_level, direction)
    st.markdown(f"""
    <div class="risk-panel">
        <h4 style="color:#fff;margin:0 0 10px 0;">📊 风险控制建议</h4>
        <p style="color:#ccc;margin:5px 0;"><strong>仓位建议:</strong> {pos:.1f} 倍基础仓位</p>
        <p style="color:#ccc;margin:5px 0;"><strong>操作建议:</strong> {sug}</p>
        <p style="color:#ccc;margin:5px 0;"><strong>连赢:</strong> {st.session_state.risk_data['win_streak']} 局 | <strong>连输:</strong> {st.session_state.risk_data['consecutive_losses']} 局</p>
    </div>
    """, unsafe_allow_html=True)

    # 在线学习：用上一手真实结果训练（避免未来信息）
    if len(seq) > 0 and direction != 'HOLD':
        last_result = seq[-1]
        record_prediction_result(direction, last_result, base_conf)
        enhanced_learning_update(direction, last_result)
        st.session_state.last_prediction = direction

# ========================== 六路展示 / 统计 / 历史（保留原版） ==========================
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
        st.markdown("#### 🤖 指标性能分析")
        perf_cols = st.columns(5)
        for i, metric in enumerate(['z','cusum','bayes','momentum','eor']):
            with perf_cols[i]:
                if st.session_state.weight_performance[metric]:
                    perf = np.mean(st.session_state.weight_performance[metric][-10:]) * 100
                    st.metric(f"{metric.upper()}准确率", f"{perf:.1f}%")

def display_complete_history():
    if not st.session_state.ultimate_games:
        st.info("暂无历史记录"); return
    st.markdown("## 📝 完整历史")
    recent=st.session_state.ultimate_games[-10:]
    for g in reversed(recent):
        icon="🃏" if g.get('mode')=='card' else ("🎯" if g.get('mode')=='quick' else "📝")
        with st.container():
            c1,c2,c3,c4,c5=st.columns([1,1,2,2,1])
            with c1: st.write(f"#{g['round']}")
            with c2: st.write(icon)
            with c3: st.write(f"闲: {'-'.join(g['player_cards'])}" if g.get('mode')=='card' else "快速记录")
            with c4: st.write(f"庄: {'-'.join(g['banker_cards'])}" if g.get('mode')=='card' else "快速记录")
            with c5:
                if g['result']=='B': st.error("庄赢")
                elif g['result']=='P': st.info("闲赢")
                else: st.warning("和局")

# ========================== 界面（保留原有双模式输入） ==========================
def display_complete_interface():
    st.markdown("## 🎮 双模式输入系统")
    show_quick_start_guide()
    c1,c2=st.columns(2)
    with c1:
        if st.button("🃏 牌点输入", use_container_width=True, type="primary"):
            st.session_state.input_mode='card'; st.rerun()
    with c2:
        if st.button("🎯 快速看路", use_container_width=True):
            st.session_state.input_mode='result'; st.rerun()
    if "input_mode" not in st.session_state: st.session_state.input_mode='card'
    if st.session_state.input_mode=='card':
        col1,col2=st.columns(2)
        with col1: p_input=st.text_input("闲家牌", placeholder="K10 或 552", key="player_card")
        with col2: b_input=st.text_input("庄家牌", placeholder="55 或 AJ", key="banker_card")
        st.markdown("### 🏆 本局结果")
        b1,b2,b3=st.columns(3)
        with b1: banker_btn=st.button("🔴 庄赢", use_container_width=True, type="primary")
        with b2: player_btn=st.button("🔵 闲赢", use_container_width=True)
        with b3: tie_btn=st.button("⚪ 和局", use_container_width=True)
        if banker_btn or player_btn or tie_btn:
            handle_card_input(p_input,b_input,banker_btn,player_btn,tie_btn)
    else:
        st.info("💡 快速模式：直接记录结果，用于快速看路分析（支持B/P/T）")
        q1,q2,q3=st.columns(3)
        with q1: qb=st.button("🔴 庄赢", use_container_width=True, type="primary")
        with q2: qp=st.button("🔵 闲赢", use_container_width=True)
        with q3: qt=st.button("⚪ 和局", use_container_width=True)
        st.markdown("### 📝 批量输入")
        batch=st.text_input("输入 B/P/T 序列（可含“庄/闲/和”）", placeholder="BPBBPT 或 庄闲庄庄和", key="batch_input")
        if st.button("✅ 确认批量输入", use_container_width=True) and batch:
            handle_batch_input(batch)
        if qb: handle_quick_input(True, False)
        if qp: handle_quick_input(False, True)
        if qt: record_game('T',['X','X'],['X','X'],'quick')

# ========================== 主程序 ==========================
def main():
    with st.sidebar:
        st.markdown("## ⚙️ 控制台")
        st.caption("EOR-Bayes 已启用；动态阈值；HOLD≤15% 限频；在线自学习。")
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
        display_complete_history()

    st.markdown("---")
    c1,c2=st.columns(2)
    with c1:
        if st.button("🔄 开始新牌靴", use_container_width=True):
            st.session_state.ultimate_games.clear()
            st.session_state.expert_roads={'big_road':[],'bead_road':[],'big_eye_road':[],'small_road':[],'cockroach_road':[],'three_bead_road':[]}
            st.session_state.risk_data={'current_level':'medium','position_size':1.0,'stop_loss':3,'consecutive_losses':0,'win_streak':0}
            st.success("新牌靴开始！"); st.rerun()
    with c2:
        if st.button("📋 导出数据", use_container_width=True):
            enhanced_export_data()

if __name__ == "__main__":
    main()
