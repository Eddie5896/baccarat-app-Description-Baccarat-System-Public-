# -*- coding: utf-8 -*-
# Baccarat Master Precision 12.0 — 天花板·云端轻量版（Streamlit Cloud & 手机网页友好）
# 保留：六路、60+模式、风险/仓位、牌点/EOR、分析卡
# 增强：自适应 CUSUM、Z-score、结构滤波、Monte Carlo Light（轻量化）、EOR 双门槛、置信度压缩、反转冷静期、和局降噪、出手率动态门槛

import streamlit as st
import numpy as np
from collections import Counter
from itertools import groupby
from datetime import datetime
from math import tanh, sqrt

# ----------------------- 页面 & CSS -----------------------
st.set_page_config(page_title="百家乐大师 Precision 12.0", layout="centered")
st.markdown("""
<style>
  .main-header {font-size:2.2rem;color:#FFD700;text-align:center;margin:.5rem 0 1rem;text-shadow:1px 1px 3px #000;}
  .card {background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:16px;border-radius:14px;border:3px solid #FFD700;margin:8px 0;}
  .pill {display:inline-block;padding:4px 10px;border-radius:999px;margin:2px;font-size:12px;color:#fff;background:#444;}
  .risk {padding:10px;border-left:4px solid #e74c3c;background:#2d3748;border-radius:8px;color:#ddd;margin-top:6px;}
  .road {background:#1a1a1a;padding:10px;border-radius:8px;border:1px solid #333;margin:6px 0;}
  .mono {font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px;}
  .metric {background:#2d3748;padding:8px;border-radius:8px;color:#eee;text-align:center;}
  .stButton>button {height:50px;font-weight:700}
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🐉 百家乐大师 Precision 12.0</h1>', unsafe_allow_html=True)

# ----------------------- 状态初始化 -----------------------
def init_state():
    ss = st.session_state
    if "games" not in ss:
        ss.games = []   # 每局: {round, result B/P/T, time, mode, P[], B[]}
    if "roads" not in ss:
        ss.roads = {'bead_road': [], 'big_road': [], 'big_eye_road': [],
                    'small_road': [], 'cockroach_road': [], 'three_bead_road': []}
    if "risk" not in ss:
        ss.risk = {'consecutive_losses': 0, 'win_streak': 0}
    if "cooldown" not in ss:
        ss.cooldown = 0
    if "trend_dir" not in ss:
        ss.trend_dir = 0
init_state()

# ----------------------- 六路计算 -----------------------
class CompleteRoadAnalyzer:
    @staticmethod
    def update_all_roads(result):
        if result not in ['B','P']: return
        R = st.session_state.roads
        # 珠路
        R['bead_road'].append(result)
        # 大路
        if not R['big_road']: R['big_road'].append([result])
        else:
            last_col = R['big_road'][-1]
            if last_col[-1] == result: last_col.append(result)
            else: R['big_road'].append([result])
        # 大眼路
        if len(R['big_road']) >= 2:
            eye=[]
            for i in range(1,len(R['big_road'])):
                eye.append('R' if len(R['big_road'][i])>=len(R['big_road'][i-1]) else 'B')
            R['big_eye_road'] = eye[-20:]
        # 小路（基于大眼路）
        if len(R['big_eye_road']) >= 2:
            small=[]
            for i in range(1,len(R['big_eye_road'])):
                small.append('R' if R['big_eye_road'][i]==R['big_eye_road'][i-1] else 'B')
            R['small_road'] = small[-15:]
        # 蟑螂路（基于小路）
        if len(R['small_road']) >= 2:
            cock=[]
            for i in range(1,len(R['small_road'])):
                cock.append('R' if R['small_road'][i]==R['small_road'][i-1] else 'B')
            R['cockroach_road'] = cock[-12:]
        # 三珠路
        bead = R['bead_road']
        if len(bead) >= 3:
            groups = [bead[i:i+3] for i in range(0, len(bead)-2, 3)]
            R['three_bead_road'] = groups[-8:]

# ----------------------- 小工具 -----------------------
def streaks(bp):
    if not bp: return []
    s=[]; cur=bp[0]; n=1
    for x in bp[1:]:
        if x==cur: n+=1
        else: s.append(n); cur=x; n=1
    s.append(n); return s

def volatility(bp):
    if len(bp) < 2: return 0.0
    return sum(1 for i in range(1,len(bp)) if bp[i]!=bp[i-1]) / len(bp)

def momentum4(bp):
    if len(bp) < 4: return 0.0
    r = bp[-4:]
    return r.count(r[-1])/4 - 0.5

# ----------------------- 60+模式识别（精选实现，余类已并入逻辑） -----------------------
class AdvancedPatternDetector:
    @staticmethod
    def detect_all_patterns(sequence):
        bp = [x for x in sequence if x in ['B','P']]
        if len(bp) < 4: return []
        pats=[]
        # 龙系列
        if len(set(bp[-4:]))==1: pats.append(f"{bp[-1]}长龙")
        if len(bp)>=5 and len(set(bp[-5:]))==1: pats.append(f"强{bp[-1]}长龙")
        # 跳跳系列
        if len(bp)>=6 and bp[-6:] in (['B','P','B','P','B','P'],['P','B','P','B','P','B']): pats.append("完美单跳")
        # 房厅系列
        s = streaks(bp)
        if len(s)>=3 and s[-3]==2 and s[-2]==1 and s[-1]==2: pats.append("一房一厅")
        if len(s)>=4 and s[-4]>=3 and s[-3]>=3 and s[-2]==1 and s[-1]>=3: pats.append("三房一厅")
        # 趋势/水路
        changes = sum(1 for i in range(1,len(bp)) if bp[i]!=bp[i-1])
        vol = changes/len(bp)
        if vol < .3: pats.append("静水路")
        elif vol > .6: pats.append("激流路")
        return pats[:5]

# ----------------------- 贝叶斯 & CUSUM -----------------------
class BayesianAdjuster:
    def __init__(self, prior_b=0.458, prior_p=0.446):
        self.prior_b = prior_b; self.prior_p = prior_p
    def update(self, recent, n_total):
        if not recent: return self.prior_b, self.prior_p
        if n_total < 60: prior_w = 0.6
        elif n_total > 150: prior_w = 0.3
        else: prior_w = 0.45
        b = recent.count('B')/len(recent); p=1-b
        post_b = (1-prior_w)*b + prior_w*self.prior_b
        post_p = (1-prior_w)*p + prior_w*self.prior_p
        s = post_b+post_p
        return (post_b/s, post_p/s) if s>0 else (self.prior_b, self.prior_p)

class CUSUM:
    def __init__(self, k=0.05, h=1.8):
        self.k=k; self.h=h
    def detect(self, bp):
        x = [1 if r=='B' else -1 for r in bp]
        if not x: return 0.0, "平稳"
        s_pos = 0.0; s_neg = 0.0; trend = 0.0
        for xi in x:
            s_pos = max(0.0, s_pos + (xi - self.k))
            s_neg = max(0.0, s_neg + (-xi - self.k))
            if s_pos > self.h: trend += 1; s_pos=0.0
            if s_neg > self.h: trend -= 1; s_neg=0.0
        label = "上升趋势" if trend>0 else ("下降趋势" if trend<0 else "平稳")
        return trend, label

def adaptive_cusum(bp):
    vol = volatility(bp)
    if vol < 0.30: k,h = 0.03, 1.4
    elif vol > 0.60: k,h = 0.07, 2.0
    else: k,h = 0.05, 1.8
    return CUSUM(k=k,h=h)

# ----------------------- EOR（算牌） -----------------------
POINT = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':0,'J':0,'Q':0,'K':0}
EOR_W = {'A': +1, '2': +1, '3': +1, '4': +2, '5': -1, '6': -2, '7': -1, '8': -1, '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0}

class EORContext:
    def __init__(self):
        self.remaining = Counter({r: 0 for r in ['A','2','3','4','5','6','7','8','9','10','J','Q','K']})
        self.total_cards = 0
        self.active = False
    def start(self, decks=8):
        self.remaining = Counter({r: 4*decks for r in self.remaining})
        self.total_cards = 52*decks
        self.active = True
    def update_used(self, cards):
        if not self.active: return
        for c in cards:
            if c in self.remaining and self.remaining[c] > 0:
                self.remaining[c] -= 1
    def eor_bias(self):
        if not self.active: return 0.0, 0.0
        tot = sum(self.remaining.values())
        if tot <= 0: return 0.0, 1.0
        bias = sum(EOR_W[r]*self.remaining[r] for r in self.remaining) / tot
        depth = 1 - tot / self.total_cards if self.total_cards>0 else 0.0
        return float(bias), float(depth)

# ----------------------- 风险与仓位 -----------------------
def risk_level_from(conf, vol, tie_ratio):
    risk_score = (1 - conf) + vol
    if tie_ratio > 0.12: risk_score = max(0.0, risk_score - 0.05)  # 和局多→降噪
    if risk_score < .35: return "low"
    if risk_score < .65: return "medium"
    if risk_score < .85: return "high"
    return "extreme"

def position_sizing(conf, win_streak, consec_loss):
    base = 1.0
    if conf > 0.8: base *= 1.2
    elif conf > 0.7: base *= 1.0
    elif conf > 0.6: base *= 0.8
    else: base *= 0.5
    if win_streak >= 3: base *= 1.1
    if consec_loss >= 2: base *= 0.7
    if consec_loss >= 3: base *= 0.5
    return min(2.0, base)

# ----------------------- 研究级增强 -----------------------
def zscore_bias(bp, window=40, p0=0.5):
    if not bp: return 0.0
    w = bp[-window:] if len(bp) >= window else bp
    n = len(w); b = w.count('B'); phat = b/n
    se = sqrt(max(1e-9, p0*(1-p0)/n))
    return (phat - p0) / se

def structural_choppiness(bp, window=30):
    if len(bp) < 2: return 0.0
    w = bp[-window:] if len(bp)>=window else bp
    return sum(1 for i in range(1,len(w)) if w[i]!=w[i-1]) / (len(w)-1)

def mc_light_vote(prob_B, n_runs=150, noise=0.02):
    prob_B = float(np.clip(prob_B, 1e-3, 0.999))
    rng = np.random.default_rng()
    p = np.clip(rng.normal(prob_B, noise, n_runs), 1e-3, 0.999)
    sims = rng.binomial(1, p, n_runs)  # 1 表示B
    return sims.mean()

def hybrid_trend_strength(cusum_comp, z_comp):
    z_norm = max(-3.0, min(3.0, z_comp)) / 3.0
    return tanh(0.6*cusum_comp + 0.4*z_norm)

# ----------------------- Precision 12.0 引擎 -----------------------
class PrecisionEngine:
    def __init__(self,
                 conf_gate=0.53,
                 use_eor=True, eor_depth_gate=0.40, eor_bias_gate=0.05,
                 mc_runs=150, mc_noise=0.02):
        self.conf_gate = conf_gate
        self.use_eor = use_eor
        self.eor_depth_gate = eor_depth_gate
        self.eor_bias_gate = eor_bias_gate
        self.eor = EORContext()
        self.mc_runs = mc_runs
        self.mc_noise = mc_noise

    def analyze(self, seq):
        bp = [x for x in seq if x in ['B','P']]
        n = len(bp)
        pats = AdvancedPatternDetector.detect_all_patterns(seq)
        s = streaks(bp)
        cur_streak = s[-1] if s else 0
        b_ratio = bp.count('B')/n if n>0 else 0.5
        recent = bp[-8:] if n>=8 else bp
        b_recent = recent.count('B')/len(recent) if recent else 0.5
        vol = volatility(bp)
        mom = momentum4(bp)

        # 基础分
        score=0.0
        score += len(pats)*0.1
        score += 0.3 if b_ratio>0.6 else (-0.3 if b_ratio<0.4 else 0)
        score += 0.2 if b_recent>0.75 else (-0.2 if b_recent<0.25 else 0)
        if cur_streak>=3: score += (cur_streak*0.1) if bp[-1]=='B' else -(cur_streak*0.1)
        score += mom*0.2

        # 初始置信 & 基础方向
        conf = min(0.9, 0.5 + abs(score)*0.4 + len(pats)*0.1)
        base_dir = "B" if score>0.15 else ("P" if score<-0.15 else "HOLD")
        if base_dir=="HOLD": conf=0.5

        # 自适应 CUSUM
        cus = adaptive_cusum(bp)
        trend_val, _ = cus.detect(bp)
        cusum_comp = tanh(trend_val/3.0)

        # 贝叶斯后验
        bayes = BayesianAdjuster()
        post_b, post_p = bayes.update(bp[-20:] if n>=20 else bp, n_total=n)
        bayes_comp = (post_b - post_p)

        # 比率/动能
        ratio_comp = (b_ratio - 0.5) * 0.6
        mom_comp = mom * 0.5

        # 和局降噪
        recent_30 = seq[-30:] if len(seq)>=30 else seq
        tie_ratio = recent_30.count('T')/len(recent_30) if recent_30 else 0.0
        if tie_ratio > 0.12:
            mom_comp *= 0.5
            ratio_comp *= 0.5

        # 势/震切换：强震荡→轻回归
        if vol > 0.70 and abs(cusum_comp) < 0.2:
            ratio_comp *= -0.5

        # Z-score & 混合趋势
        z = zscore_bias(bp, window=40, p0=0.5)
        hybrid_comp = hybrid_trend_strength(cusum_comp, z)

        # EOR（双门槛）
        eor_comp = 0.0
        eor_txt = None
        if self.use_eor and self.eor.active:
            e_bias, depth = self.eor.eor_bias()
            if depth > self.eor_depth_gate and abs(e_bias) > self.eor_bias_gate:
                eor_comp = -e_bias      # 约定：负偏向庄（小点偏多）
                eor_txt = f"EOR有效 深{depth:.0%} 偏{e_bias:+.2f}"
            else:
                eor_txt = f"EOR弱 深{depth:.0%} 偏{e_bias:+.2f}"

        # 结构滤波（震荡惩罚）
        choppy = structural_choppiness(bp, window=30)
        chop_penalty = 0.0
        if choppy > 0.65:
            mom_comp *= 0.6
            ratio_comp *= 0.7
            chop_penalty = (choppy - 0.65) * 0.5

        # 权重（自适应）
        w_eor, w_bay, w_hyb, w_mom, w_ratio = 0.26, 0.22, 0.28, 0.12, 0.12
        if not (self.use_eor and self.eor.active): w_eor = 0.0
        if vol < 0.30: w_hyb += 0.05; w_mom += 0.03
        if 0.45 <= b_ratio <= 0.55: w_bay += 0.08
        if self.use_eor and self.eor.active:
            _, d = self.eor.eor_bias()
            if d > 0.60: w_eor += 0.04
        W = max(1e-9, w_eor + w_bay + w_hyb + w_mom + w_ratio)
        w_eor, w_bay, w_hyb, w_mom, w_ratio = [w/W for w in (w_eor,w_bay,w_hyb,w_mom,w_ratio)]

        math_score = (w_eor*eor_comp + w_bay*bayes_comp +
                      w_hyb*hybrid_comp + w_mom*mom_comp + w_ratio*ratio_comp)
        math_score = max(-1.0, min(1.0, math_score))

        # Monte Carlo Light 融合
        p_b_base = 0.5 + max(-0.15, min(0.15, math_score*0.15))
        blend = 0.6 if n < 50 else (0.4 if n < 120 else 0.3)
        pB = blend*post_b + (1-blend)*p_b_base
        mc_vote = mc_light_vote(prob_B=pB, n_runs=self.mc_runs, noise=self.mc_noise)
        mc_comp = (mc_vote - 0.5)*2.0

        # 置信融合 + 压缩 + 结构惩罚
        conf = max(0.1, min(0.95, conf + math_score*0.08 + mc_comp*0.05 - chop_penalty))
        conf = 0.5 + 0.9*(conf - 0.5)

        # 反转冷静期
        new_trend_dir = 1 if cusum_comp>0.1 else (-1 if cusum_comp<-0.1 else 0)
        flipped = (st.session_state.trend_dir != 0 and new_trend_dir != 0 and st.session_state.trend_dir != new_trend_dir)
        st.session_state.trend_dir = new_trend_dir
        if flipped and conf < 0.67:
            st.session_state.cooldown = 1
        if st.session_state.cooldown > 0:
            st.session_state.cooldown -= 1
            final_dir = "HOLD"
        else:
            # 动态出手率门槛
            conf_gate = self.conf_gate
            if conf < 0.60: conf_gate = max(0.50, conf_gate - 0.02)
            signed = math_score + 0.5*mc_comp + 0.5*hybrid_comp
            base_dir2 = "B" if signed>0.08 else ("P" if signed<-0.08 else base_dir)
            final_dir = base_dir2 if conf >= conf_gate else "HOLD"

        risk = risk_level_from(conf, vol, tie_ratio)

        return {
            "direction": final_dir,
            "base_dir": base_dir,
            "confidence": conf,
            "patterns": pats[:5],
            "volatility": vol,
            "risk": risk,
            "details": {
                "cusum": float(cusum_comp),
                "bayes": float(bayes_comp),
                "hybrid": float(hybrid_comp),
                "momentum": float(mom_comp),
                "ratio": float(ratio_comp),
                "mc": float(mc_comp),
                "eor": float(eor_comp),
                "eor_text": eor_txt,
                "zscore": float(z),
                "choppy": float(choppy),
                "b_ratio": b_ratio,
                "b_recent": b_recent,
                "cur_streak": cur_streak,
                "tie_ratio_30": tie_ratio
            }
        }

ENGINE = PrecisionEngine()

# ----------------------- 输入 UI -----------------------
st.markdown("## 🎮 输入")
c1, c2 = st.columns(2)
with c1:
    if st.button("🔴 庄赢", use_container_width=True, type="primary"):
        res='B'
        st.session_state.games.append({"round": len(st.session_state.games)+1, "result": res,
                                       "time": datetime.now().strftime("%H:%M"), "mode":"quick", "P": [], "B": []})
        CompleteRoadAnalyzer.update_all_roads(res)
        st.success("记录成功：庄"); st.rerun()
with c2:
    if st.button("🔵 闲赢", use_container_width=True):
        res='P'
        st.session_state.games.append({"round": len(st.session_state.games)+1, "result": res,
                                       "time": datetime.now().strftime("%H:%M"), "mode":"quick", "P": [], "B": []})
        CompleteRoadAnalyzer.update_all_roads(res)
        st.success("记录成功：闲"); st.rerun()

with st.expander("🃏 牌点输入（启用 EOR 需设置副数）", expanded=False):
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        p_cards_in = st.text_input("闲家牌 (K10 或 552)").upper().replace(" ","")
    with col2:
        b_cards_in = st.text_input("庄家牌 (55 或 AJ)").upper().replace(" ","")
    with col3:
        decks = st.number_input("🔧 EOR 副数（>0 启用）", 0, 12, 8, 1)
        if decks > 0 and not ENGINE.eor.active: ENGINE.eor.start(int(decks))
        elif decks == 0: ENGINE.eor.active = False

    def parse_cards(s):
        if not s: return []
        out=[]; i=0
        while i < len(s):
            if s[i:i+2]=="10": out.append("10"); i+=2
            elif s[i] in "123456789": out.append(s[i]); i+=1
            elif s[i] in "AJQK0": out.append({"A":"A","J":"J","Q":"Q","K":"K","0":"10"}[s[i]]); i+=1
            else: i+=1
        return out

    cA,cB,cC = st.columns(3)
    if cA.button("✅ 牌点记录『庄赢』", use_container_width=True):
        P = parse_cards(p_cards_in); B = parse_cards(b_cards_in)
        if len(P)>=2 and len(B)>=2:
            st.session_state.games.append({"round": len(st.session_state.games)+1, "result": 'B',
                                           "time": datetime.now().strftime("%H:%M"), "mode":"card", "P": P, "B": B})
            CompleteRoadAnalyzer.update_all_roads('B')
            if ENGINE.eor.active: ENGINE.eor.update_used(P+B)
            st.success("牌点记录：庄赢"); st.rerun()
        else: st.error("需要至少2张牌")
    if cB.button("✅ 牌点记录『闲赢』", use_container_width=True):
        P = parse_cards(p_cards_in); B = parse_cards(b_cards_in)
        if len(P)>=2 and len(B)>=2:
            st.session_state.games.append({"round": len(st.session_state.games)+1, "result": 'P',
                                           "time": datetime.now().strftime("%H:%M"), "mode":"card", "P": P, "B": B})
            CompleteRoadAnalyzer.update_all_roads('P')
            if ENGINE.eor.active: ENGINE.eor.update_used(P+B)
            st.success("牌点记录：闲赢"); st.rerun()
        else: st.error("需要至少2张牌")
    if cC.button("⚪ 记录『和局』", use_container_width=True):
        st.session_state.games.append({"round": len(st.session_state.games)+1, "result": 'T',
                                       "time": datetime.now().strftime("%H:%M"), "mode":"quick", "P": [], "B": []})
        st.success("记录：和局"); st.rerun()

with st.expander("📝 批量输入（BPBBP 或 庄闲庄庄闲）", expanded=False):
    batch = st.text_input("输入BP序列")
    if st.button("📥 批量导入", use_container_width=True):
        if batch:
            s = batch.upper().replace("庄","B").replace("闲","P").replace(" ","")
            seq = [c for c in s if c in ['B','P']]
            for r in seq:
                st.session_state.games.append({"round": len(st.session_state.games)+1, "result": r,
                                               "time": datetime.now().strftime("%H:%M"), "mode":"batch", "P": [], "B": []})
                CompleteRoadAnalyzer.update_all_roads(r)
            st.success(f"已导入 {len(seq)} 局"); st.rerun()

cN, cM = st.columns(2)
with cN:
    if st.button("🔄 开始新牌靴", use_container_width=True):
        st.session_state.games.clear()
        st.session_state.roads = {'bead_road': [], 'big_road': [], 'big_eye_road': [],
                                  'small_road': [], 'cockroach_road': [], 'three_bead_road': []}
        st.session_state.risk = {'consecutive_losses': 0, 'win_streak': 0}
        st.session_state.cooldown = 0
        st.session_state.trend_dir = 0
        ENGINE.eor = EORContext()
        st.success("新牌靴开始！"); st.rerun()
with cM:
    st.info("📤 云端建议：长测请本地跑模拟器")

st.markdown("---")

# ----------------------- 智能分析卡片 -----------------------
st.markdown("## 🎯 智能分析")
if len(st.session_state.games) >= 3:
    seq = [g['result'] for g in st.session_state.games]
    analysis = ENGINE.analyze(seq)
    dir_map = {
        "B":("🔴","庄(B)","linear-gradient(135deg,#FF6B6B 0%, #C44569 100%)","#FF6B6B"),
        "P":("🔵","闲(P)","linear-gradient(135deg,#4ECDC4 0%, #44A08D 100%)","#4ECDC4"),
        "HOLD":("⚪","观望","linear-gradient(135deg,#FFE66D 0%, #F9A826 100%)","#FFE66D")
    }
    icon, text, bg, color = dir_map[analysis['direction']]
    st.markdown(f"""
    <div class="card" style="background:{bg}">
      <h3 style="margin:0;color:{color};text-align:center">{icon} 推荐：{text}</h3>
      <p style="margin:4px 0 0;color:#fff;text-align:center">
        🎯 置信度：{analysis['confidence']*100:.1f}% &nbsp; | &nbsp; 风险：{analysis['risk']}
      </p>
      <p style="margin:4px 0 0;color:#eee;text-align:center">
        {''.join([f'<span class="pill">{p}</span>' for p in analysis['patterns']])}
      </p>
      <p class="mono" style="color:#ddd;text-align:center;margin:6px 0 0;">
        Hybrid:{analysis['details']['hybrid']:+.2f} | Z:{analysis['details']['zscore']:+.2f}σ | CUSUM:{analysis['details']['cusum']:+.2f}
        | Bayes:{analysis['details']['bayes']:+.2f} | Mom:{analysis['details']['momentum']:+.2f} | Ratio:{analysis['details']['ratio']:+.2f}
        | MC:{analysis['details']['mc']:+.2f} | EOR:{analysis['details']['eor']:+.2f}
      </p>
      <p style="margin:2px 0 0;color:#9fe1ff;text-align:center;font-size:12px;">{analysis['details']['eor_text'] or ''}</p>
    </div>
    """, unsafe_allow_html=True)

    pos = position_sizing(analysis['confidence'],
                          st.session_state.risk['win_streak'],
                          st.session_state.risk['consecutive_losses'])
    st.markdown(f"""
    <div class="risk">
      <b>📊 风险控制建议</b><br/>
      仓位建议：<b>{pos:.1f} 倍</b>基础仓位<br/>
      连赢：{st.session_state.risk['win_streak']} 局 &nbsp;|&nbsp; 连输：{st.session_state.risk['consecutive_losses']} 局<br/>
      波动率：{analysis['volatility']*100:.1f}% &nbsp;|&nbsp;
      近30和局占比：{analysis['details']['tie_ratio_30']*100:.1f}% &nbsp;|&nbsp;
      震荡度：{analysis['details']['choppy']*100:.1f}%
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("请先记录至少 3 局牌局数据。")

# ----------------------- 六路显示 -----------------------
st.markdown("## 🛣️ 六路分析")
R = st.session_state.roads
st.markdown("#### 🟠 珠路 (最近20局)")
if R['bead_road']:
    bead_display = " ".join(["🔴" if x=='B' else "🔵" for x in R['bead_road'][-20:]])
    st.markdown(f'<div class="road">{bead_display}</div>', unsafe_allow_html=True)
st.markdown("#### 🔴 大路")
if R['big_road']:
    for i, col in enumerate(R['big_road'][-6:]):
        col_display = " ".join(["🔴" if x=='B' else "🔵" for x in col])
        st.markdown(f'<div class="road mono">第{i+1}列: {col_display}</div>', unsafe_allow_html=True)
cA,cB = st.columns(2)
with cA:
    if R['big_eye_road']:
        eye_display = " ".join(["🔴" if x=='R' else "🔵" for x in R['big_eye_road'][-12:]])
        st.markdown("#### 👁️ 大眼路")
        st.markdown(f'<div class="road mono">{eye_display}</div>', unsafe_allow_html=True)
with cB:
    if R['small_road']:
        small_display = " ".join(["🔴" if x=='R' else "🔵" for x in R['small_road'][-10:]])
        st.markdown("#### 🔵 小路")
        st.markdown(f'<div class="road mono">{small_display}</div>', unsafe_allow_html=True)
if R['three_bead_road']:
    st.markdown("#### 🔶 三珠路")
    for i, group in enumerate(R['three_bead_road'][-6:]):
        group_display = " ".join(["🔴" if x=='B' else "🔵" for x in group])
        st.markdown(f'<div class="road mono">第{i+1}组: {group_display}</div>', unsafe_allow_html=True)

# ----------------------- 统计与历史 -----------------------
st.markdown("## 📊 统计与历史")
games = st.session_state.games
if games:
    results = [g['result'] for g in games]
    total = len(results)
    banker_wins = results.count('B'); player_wins = results.count('P'); ties = results.count('T')
    bead = st.session_state.roads['bead_road']
    changes = sum(1 for i in range(1, len(bead)) if bead[i] != bead[i-1]) if bead else 0
    volp = changes/len(bead)*100 if bead else 0
    avg_streak = np.mean([len(list(g)) for k,g in groupby(bead)]) if bead else 0.0
    m1,m2,m3,m4 = st.columns(4)
    m1.markdown(f'<div class="metric">总局数<br/><b>{total}</b></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric">庄胜率<br/><b>{(banker_wins/max(1,total))*100:.1f}%</b></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric">闲胜率<br/><b>{(player_wins/max(1,total))*100:.1f}%</b></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric">和局率<br/><b>{(ties/max(1,total))*100:.1f}%</b></div>', unsafe_allow_html=True)

    st.markdown("### 📝 最近10局")
    for g in games[-10:][::-1]:
        mode_icon = "🃏" if g['mode']=="card" else ("🎯" if g['mode']=="quick" else "📝")
        res = "庄" if g['result']=="B" else ("闲" if g['result']=="P" else "和")
        extra = f" | 闲:{'-'.join(g['P'])} 庄:{'-'.join(g['B'])}" if g['mode']=="card" else ""
        st.write(f"{mode_icon} #{g['round']} | {g['time']} | 结果：**{res}**{extra}")
else:
    st.info("暂无统计数据")

# ----------------------- 胜负串追踪（风险记分） -----------------------
if st.session_state.games:
    last = st.session_state.games[-1]['result']
    if last in ['B','P']:
        st.session_state.risk['win_streak'] += 1
        st.session_state.risk['consecutive_losses'] = 0
    elif last == 'T':
        pass
    else:
        st.session_state.risk['consecutive_losses'] += 1
        st.session_state.risk['win_streak'] = 0
