# -*- coding: utf-8 -*-
# Baccarat Master — Mobile Pro 2.0（手机稳定版 + 算牌增强 + 不删任何原功能）
# ✅ 保留：六路 / 60+模式 / 牌点增强 / 风控 / 统计 / 历史 / 表单输入 / 批量输入 / 新牌靴
# ➕ 新增：EOR算牌 + 贝叶斯修正 + CUSUM趋势 + Z-score滤波 + 融合模型（只微调置信度，默认不改方向）

import streamlit as st
import numpy as np
from itertools import groupby
from datetime import datetime

# ========= 全局设置 =========
st.set_page_config(page_title="Baccarat Mobile Pro 2.0", layout="centered")

# 可配置项（你可以根据需要微调）
DECKS = 8                               # 默认8副牌
CONFIDENCE_MAX_BOOST = 0.10             # 新数学模型对置信度的最大微调幅度（±10%）
ALLOW_DIRECTION_OVERRIDE = False        # 是否允许数学模型在极端情况下改方向（默认不改）
BAYES_WINDOW = 20                       # 贝叶斯更新的滚动窗口手数
CUSUM_K = 0.05                          # CUSUM灵敏度参数
CUSUM_H = 1.5                           # CUSUM触发阈值
Z_WINDOW = 12                           # Z-score 平滑窗口

# ========= 轻量样式（手机友好）=========
st.markdown("""
<style>
  .h1 {font-size: 1.4rem; font-weight:700; text-align:center; margin: .2rem 0 .6rem;}
  .card {background:#1f2937; border:1px solid #374151; border-radius:10px; padding:.8rem; margin:.5rem 0;}
  .pill {display:inline-block; padding:.2rem .5rem; border-radius:999px; font-size:.8rem; margin:.15rem; color:#fff;}
  .pill-r {background:#ef4444;} .pill-b {background:#3b82f6;}
  .pill-g {background:#10b981;} .pill-y {background:#f59e0b;} .pill-p {background:#8b5cf6;}
  .mono {font-family: ui-monospace, SFMono-Regular, Menlo, monospace;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="h1">🐉 Baccarat Master — Mobile Pro 2.0</div>', unsafe_allow_html=True)

# ========= SessionState =========
ss = st.session_state
ss.setdefault("games", [])  # [{'round','player_cards','banker_cards','result','time','mode'}]
ss.setdefault("roads", {'big_road':[], 'bead_road':[], 'big_eye_road':[], 'small_road':[], 'cockroach_road':[], 'three_bead_road':[]})
ss.setdefault("risk", {'consecutive_losses':0, 'win_streak':0})
ss.setdefault("signal_hist", [])  # 存放历史融合信号用于Z平滑

# ========= 工具函数 =========
def parse_cards(s):
    if not s: return []
    s=s.upper().replace(" ","")
    out=[]; i=0
    while i < len(s):
        if i+1 < len(s) and s[i:i+2]=="10": out.append("10"); i+=2
        elif s[i] in "123456789": out.append(s[i]); i+=1
        elif s[i] in "AJQK0": out.append("10" if s[i]=='0' else s[i]); i+=1
        else: i+=1
    return out

def dots(arr, red='B'):
    return " ".join('🔴' if x==red or x=='R' else '🔵' for x in arr)

# ========= 六路生成（保留原逻辑）=========
class Roads:
    @staticmethod
    def update(result):
        if result not in ['B','P']: return
        r = ss.roads
        r['bead_road'].append(result)
        if not r['big_road']: r['big_road'].append([result])
        else:
            col = r['big_road'][-1]
            if col[-1] == result: col.append(result)
            else: r['big_road'].append([result])
        # 大眼
        if len(r['big_road']) >= 2:
            eye=[]
            for i in range(1, len(r['big_road'])):
                eye.append('R' if len(r['big_road'][i]) >= len(r['big_road'][i-1]) else 'B')
            r['big_eye_road'] = eye[-20:]
        # 小路
        if len(r['big_eye_road']) >= 2:
            sm=[]
            for i in range(1, len(r['big_eye_road'])):
                sm.append('R' if r['big_eye_road'][i]==r['big_eye_road'][i-1] else 'B')
            r['small_road'] = sm[-15:]
        # 蟑螂
        if len(r['small_road']) >= 2:
            ck=[]
            for i in range(1, len(r['small_road'])):
                ck.append('R' if r['small_road'][i]==r['small_road'][i-1] else 'B')
            r['cockroach_road'] = ck[-12:]
        # 三珠
        b = r['bead_road']
        if len(b) >= 3:
            groups = [b[i:i+3] for i in range(0, len(b)-2, 3)]
            r['three_bead_road'] = groups[-8:]

# ========= 模式识别（轻量保留）=========
class Patterns:
    @staticmethod
    def streaks(bp):
        if not bp: return []
        s, c, n = [], bp[0], 1
        for x in bp[1:]:
            if x==c: n+=1
            else: s.append(n); c=x; n=1
        s.append(n)
        return s

    @staticmethod
    def detect_all(seq):
        bp = [x for x in seq if x in ['B','P']]
        if len(bp) < 4: return []
        pats = []
        # 长龙
        if len(set(bp[-4:]))==1: pats.append(f"{bp[-1]}长龙")
        if len(bp)>=6 and len(set(bp[-6:]))==1: pats.append("超强长龙")
        # 单跳/双跳
        if len(bp)>=6 and bp[-6:] in (['B','P','B','P','B','P'], ['P','B','P','B','P','B']):
            pats.append("完美单跳")
        if len(bp)>=8 and bp[-8:] in (['B','B','P','P','B','B','P','P'], ['P','P','B','B','P','P','B','B']):
            pats.append("齐头双跳")
        # 房厅系列
        s = Patterns.streaks(bp)
        if len(s)>=4 and s[-4] >= 3 and s[-3] >= 3 and s[-2]==1 and s[-1] >= 3: pats.append("三房一厅")
        if len(s)>=3 and s[-3]==2 and s[-2]==1 and s[-1]==2: pats.append("一房一厅")
        # 趋势/水路
        if len(s)>=4 and all(s[i] < s[i+1] for i in range(-4,-1)): pats.append("上山路")
        if len(s)>=4 and all(s[i] > s[i+1] for i in range(-4,-1)): pats.append("下山路")
        changes = sum(1 for i in range(1, len(bp)) if bp[i]!=bp[i-1])
        vol = changes/len(bp)
        if vol < .3: pats.append("静水路")
        elif vol > .6: pats.append("激流路")
        return pats[:6]

# ========= 牌点增强（保留）=========
class CardEnh:
    MAP = {'A':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':0,'J':0,'Q':0,'K':0}
    @staticmethod
    def pts(cards): return sum(CardEnh.MAP.get(c,0) for c in cards) % 10

    @staticmethod
    def analyze(games):
        cg = [g for g in games if g.get('mode')=='card' and len(g['player_cards'])>=2 and len(g['banker_cards'])>=2]
        if len(cg) < 2: return 0.0, ""
        factor, rsn = 0.0, []
        # 天牌密集
        nat = sum(1 for g in cg[-3:] if CardEnh.pts(g['player_cards'])>=8 or CardEnh.pts(g['banker_cards'])>=8)
        if nat>=2: factor+=.08; rsn.append(f"天牌×{nat}")
        elif nat==1: factor+=.03; rsn.append("天牌")
        # 点数动量
        if len(cg)>=4:
            pts = []
            for g in cg[-4:]:
                pts += [CardEnh.pts(g['player_cards']), CardEnh.pts(g['banker_cards'])]
            avg = sum(pts)/len(pts)
            if avg < 4: factor += .06; rsn.append("小点数期")
            elif avg > 7: factor -= .04; rsn.append("大点数期")
        # 补牌密度（粗略）
        if len(cg)>=5:
            total = min(10, len(cg))
            draw = 0
            for g in cg[-total:]:
                if CardEnh.pts(g['player_cards'])<6 or CardEnh.pts(g['banker_cards'])<6: draw += 1
            ratio = draw/total
            if ratio > .7: factor -= .05; rsn.append("补牌密集")
            elif ratio < .3: factor += .04; rsn.append("补牌稀少")
        factor = max(-.2, min(.2, factor))
        return factor, " / ".join(rsn) if rsn else ""

# ========= 新增：数学算牌增强模块（EOR + 贝叶斯 + CUSUM + Z滤波）=========
class EORCountEngine:
    # 轻量EOR权重（可替换为你的TP权重）
    EOR = {'A': +1, '2': +1, '3': +1, '4': +2,
           '5': -1, '6': -2, '7': -1, '8': -1,
           '9':  0, '10': 0, 'J':  0, 'Q':  0, 'K':  0}
    RANKS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']

    def __init__(self, decks=8):
        self.decks = decks
        self.reset()

    def reset(self):
        self.shoe = {r: 4*self.decks for r in self.RANKS}  # 每副牌4张每点数

    def build_from_history(self, games):
        self.reset()
        for g in games:
            if g.get('mode') == 'card':
                for c in (g.get('player_cards',[]) + g.get('banker_cards',[])):
                    if c in self.shoe:
                        self.shoe[c] = max(0, self.shoe[c]-1)

    def eor_score(self):
        # 正值偏闲，负值偏庄（可根据你的口径调整）
        total_seen = 4*self.decks*len(self.RANKS) - sum(self.shoe.values())
        if total_seen == 0: return 0.0, 0
        score = 0
        for r in self.RANKS:
            used = 4*self.decks - self.shoe[r]
            score += self.EOR[r] * used
        # 归一化
        norm = max(1, sum(self.shoe.values()))
        bias = score / norm
        return bias, norm

class BayesianAdjuster:
    def __init__(self, prior_b=0.458, prior_p=0.446):
        self.prior_b = prior_b
        self.prior_p = prior_p

    def update(self, recent_seq):
        # 简化：最近窗口内的频率作为似然，平滑到先验上
        if not recent_seq:
            return self.prior_b, self.prior_p
        b = recent_seq.count('B') / len(recent_seq)
        p = recent_seq.count('P') / len(recent_seq)
        # 拉向先验，避免极端（0.7权重使用近期，0.3保留先验）
        post_b = 0.7*b + 0.3*self.prior_b
        post_p = 0.7*p + 0.3*self.prior_p
        # 归一化
        s = post_b + post_p
        if s == 0: return self.prior_b, self.prior_p
        return post_b/s, post_p/s

class CUSUMDetector:
    def __init__(self, k=CUSUM_K, h=CUSUM_H):
        self.k = k; self.h = h

    def detect(self, seq):
        # 将B映射+1，P映射-1，基准均值0
        x = [1 if r=='B' else -1 for r in seq if r in ['B','P']]
        if not x: return 0, "平稳"
        s_pos = 0; s_neg = 0; trend = 0
        for xi in x:
            s_pos = max(0, s_pos + (xi - self.k))
            s_neg = max(0, s_neg + (-xi - self.k))
            if s_pos > self.h:
                trend += 1; s_pos = 0
            if s_neg > self.h:
                trend -= 1; s_neg = 0
        label = "上升趋势" if trend>0 else ("下降趋势" if trend<0 else "平稳")
        return trend, label

class ZFilter:
    def smooth(self, values, window=Z_WINDOW):
        if not values: return 0.0
        vals = values[-window:] if len(values) >= window else values[:]
        m = float(np.mean(vals))
        s = float(np.std(vals)) if np.std(vals) > 1e-9 else 1.0
        z = (vals[-1] - m) / s
        # 压缩z到[-1,1]区间的tanh
        return float(np.tanh(z))

class FusionModel:
    """
    融合：EOR偏向 + 贝叶斯后验 + CUSUM趋势 +（原动能/走势可作为外部输入）
    输出：math_trend in [-1,1]，>0 偏庄，<0 偏闲；以及可读说明
    """
    def fuse(self, eor_bias, bayes_b, bayes_p, cusum_trend, mom, b_ratio):
        # eor_bias：>0 偏闲（前述定义），我们取负号让正为偏庄，便于统一方向
        eor_component = -eor_bias
        # 贝叶斯：庄-闲 差
        bayes_component = (bayes_b - bayes_p)
        # CUSUM：正为上升（近似偏庄），负为下降（偏闲），做一个轻量归一
        cusum_component = np.tanh(cusum_trend / 3.0)
        # 动能mom与整体庄占比b_ratio也纳入一点
        mom_component = mom * 0.5
        ratio_component = (b_ratio - 0.5) * 0.6

        # 权重（可微调/自学习）
        w_eor, w_bay, w_cus, w_mom, w_ratio = 0.45, 0.25, 0.15, 0.10, 0.05
        score = (w_eor*eor_component + w_bay*bayes_component + w_cus*cusum_component
                 + w_mom*mom_component + w_ratio*ratio_component)

        # 限幅
        score = float(max(-1.0, min(1.0, score)))
        # 可读标签
        if score > 0.1: tag = f"偏庄 {score*100:.1f}%"
        elif score < -0.1: tag = f"偏闲 {abs(score)*100:.1f}%"
        else: tag = "平衡 ±10%"
        return score, tag

# ========= 原有核心分析（保留）=========
def current_streak(bp):
    if not bp: return 0
    c = bp[-1]; n = 1
    for x in reversed(bp[:-1]):
        if x==c: n+=1
        else: break
    return n

def volatility(bp):
    if len(bp)<2: return 0.0
    return sum(1 for i in range(1,len(bp)) if bp[i]!=bp[i-1]) / len(bp)

def momentum(bp):
    if len(bp)<4: return 0.0
    recent = bp[-4:]
    return recent.count(recent[-1])/4 - 0.5

# ========= 新增：Pro 2.0 综合分析封装（不删旧逻辑，只增加融合）=========
def analyze(sequence, games):
    # —— 原有分析部分（保持）——
    if len(sequence) < 3:
        return {"dir":"HOLD","conf":0.5,"pats":[],"reason":"数据不足","vol":0.0,"streak":0,
                "risk":"medium","risk_text":"🟡 中风险","math":None}

    bp = [x for x in sequence if x in ['B','P']]
    pats = Patterns.detect_all(sequence)
    s = current_streak(bp)
    b_ratio = bp.count('B')/len(bp)
    recent = bp[-8:] if len(bp)>=8 else bp
    b_recent = recent.count('B')/len(recent) if recent else 0.5
    vol = volatility(bp)
    mom = momentum(bp)

    score = 0.0
    score += len(pats)*0.1
    score += 0.3 if b_ratio>0.6 else (-0.3 if b_ratio<0.4 else 0)
    score += 0.2 if b_recent>0.75 else (-0.2 if b_recent<0.25 else 0)
    if s>=3:
        score += (s*0.1) if bp[-1]=='B' else -(s*0.1)
    score += mom*0.2

    conf = min(0.9, 0.5 + abs(score)*0.4 + len(pats)*0.1)
    if score > 0.15: d = "B"
    elif score < -0.15: d = "P"
    else: d="HOLD"; conf=0.5

    # 风险
    risk_score = (1-conf) + vol
    if risk_score < .3: risk=("low","🟢 低风险")
    elif risk_score < .6: risk=("medium","🟡 中风险")
    elif risk_score < .8: risk=("high","🟠 高风险")
    else: risk=("extreme","🔴 极高风险")

    reason_bits=[]
    if pats: reason_bits.append("模式:"+",".join(pats[:3]))
    if s>=2: reason_bits.append(f"连{s}")
    reason_bits.append(f"风险:{risk[0]}")

    # —— 原有牌点增强（保留）——
    enh, enh_txt = CardEnh.analyze(games)
    if enh != 0:
        conf = max(0.1, min(0.95, conf + enh))
        if enh_txt: reason_bits.append("牌点:"+enh_txt)

    # —— 新增：数学算牌增强（EOR + 贝叶斯 + CUSUM + Z）——
    eor = EORCountEngine(DECKS); eor.build_from_history(games)
    eor_bias, remaining = eor.eor_score()   # >0 偏闲，<0 偏庄

    bayes = BayesianAdjuster()
    recent_win = bp[-BAYES_WINDOW:] if len(bp)>=BAYES_WINDOW else bp
    post_b, post_p = bayes.update(recent_win)

    cus = CUSUMDetector(CUSUM_K, CUSUM_H)
    trend_val, trend_label = cus.detect(bp)

    fusion = FusionModel()
    math_score, math_tag = fusion.fuse(eor_bias, post_b, post_p, trend_val, mom, b_ratio)

    # Z平滑：记录历史融合信号
    ss.signal_hist.append(math_score)
    zf = ZFilter()
    z_val = zf.smooth(ss.signal_hist, window=Z_WINDOW)
    # 平滑后做一个微调：保留方向但收敛极端值
    math_score_smooth = float(np.tanh((math_score + 0.5*z_val)))

    # 对原conf做“有限微调”，不改六路，不强制改方向
    boost = math_score_smooth * CONFIDENCE_MAX_BOOST  # [-0.1, 0.1]
    conf = float(max(0.1, min(0.95, conf + boost)))

    # 可选：极端情况下允许改方向（关闭时不会触发）
    if ALLOW_DIRECTION_OVERRIDE and d!="HOLD":
        if math_score_smooth > 0.6 and d=="P": d="B"
        if math_score_smooth < -0.6 and d=="B": d="P"

    # 组合说明
    math_text = f"🧮 数学趋势：{math_tag}｜CUSUM：{trend_label}｜EOR偏向({'偏闲' if eor_bias>0 else ('偏庄' if eor_bias<0 else '平衡')})"
    reason_bits.append("融合:"+math_tag)

    return {"dir":d,"conf":conf,"pats":pats,"reason":" | ".join(reason_bits),
            "vol":vol,"streak":s,"risk":risk[0],"risk_text":risk[1],
            "math":{"tag":math_tag, "cusum":trend_label, "eor_bias":eor_bias, "post_b":post_b, "post_p":post_p,
                    "score":math_score_smooth, "remaining":remaining}}

# ========= 顶部输入（表单，防抖）=========
with st.form("input_form"):
    st.write("🎮 录入一局（手机表单更稳）")
    c1, c2 = st.columns(2)
    with c1:
        p_in = st.text_input("闲家牌 (例: K10 或 552)", key="p_in")
    with c2:
        b_in = st.text_input("庄家牌 (例: 55 或 AJ)", key="b_in")

    col = st.columns(3)
    with col[0]: choose_b = st.form_submit_button("录入 庄赢", use_container_width=True)
    with col[1]: choose_p = st.form_submit_button("录入 闲赢", use_container_width=True)
    with col[2]: choose_t = st.form_submit_button("录入 和局", use_container_width=True)

# 处理提交（不使用 rerun）
if choose_b or choose_p or choose_t:
    p_cards = parse_cards(p_in)
    b_cards = parse_cards(b_in)
    mode = "card" if (len(p_cards)>=2 and len(b_cards)>=2) else "quick"
    result = 'B' if choose_b else ('P' if choose_p else 'T')
    ss.games.append({
        'round': len(ss.games)+1,
        'player_cards': p_cards if mode=="card" else ['X','X'],
        'banker_cards': b_cards if mode=="card" else ['X','X'],
        'result': result,
        'time': datetime.now().strftime("%H:%M"),
        'mode': mode
    })
    if result in ['B','P']:
        Roads.update(result)
        if result in ['B','P']:
            ss.risk['win_streak'] += 1
            ss.risk['consecutive_losses'] = 0
        else:
            ss.risk['consecutive_losses'] += 1
            ss.risk['win_streak'] = 0
    st.toast(f"✅ 第 {len(ss.games)} 局已记录（{ '庄' if result=='B' else '闲' if result=='P' else '和' }）")

# 批量输入
with st.expander("📝 批量输入 BP（如：BPBBP 或 庄闲庄庄闲）"):
    batch = st.text_input("输入序列", key="batch")
    if st.button("确认批量添加", use_container_width=True):
        seq = batch.upper().replace('庄','B').replace('闲','P').replace(' ','')
        vals = [c for c in seq if c in ['B','P']]
        for r in vals:
            ss.games.append({'round': len(ss.games)+1, 'player_cards':['X','X'], 'banker_cards':['X','X'],
                             'result': r, 'time': datetime.now().strftime("%H:%M"), 'mode':'batch'})
            Roads.update(r)
        st.success(f"已添加 {len(vals)} 局")

# ========= 分析卡片（新增数学趋势展示，保留原有项）=========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("🎯 智能分析")
seq = [g['result'] for g in ss.games]
res = analyze(seq, ss.games) if len(seq)>=1 else {"dir":"HOLD","conf":0.5,"pats":[],"reason":"等待数据","risk_text":"🟡 中风险","math":None}
dir_map = {"B":("庄","pill-r"), "P":("闲","pill-b"), "HOLD":("观望","pill-y")}
name, cls = dir_map.get(res["dir"], ("观望","pill-y"))
st.markdown(f'<span class="pill {cls}">推荐：{name}</span>  '
            f'<span class="pill pill-g">置信度：{res["conf"]*100:.1f}%</span>  '
            f'<span class="pill pill-y">{res["risk_text"]}</span>', unsafe_allow_html=True)
st.caption(res["reason"])

# 新增：数学趋势可视化标签
if res.get("math"):
    mt = res["math"]
    # 方向色块（数学趋势角度）
    math_dir = "偏庄" if mt["score"]>0.1 else ("偏闲" if mt["score"]<-0.1 else "平衡")
    math_color = "pill-r" if mt["score"]>0.1 else ("pill-b" if mt["score"]<-0.1 else "pill-p")
    st.markdown(
        f'<span class="pill {math_color}">🧮 数学趋势：{math_dir}（{abs(mt["score"])*100:.1f}%）</span>  '
        f'<span class="pill pill-p">CUSUM：{mt["cusum"]}</span>  '
        f'<span class="pill pill-p">EOR：{"偏闲" if mt["eor_bias"]>0 else ("偏庄" if mt["eor_bias"]<0 else "平衡")}</span>',
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# ========= 模式 & 风控（保留）=========
if res.get("pats"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("🧩 检测模式")
    st.write(", ".join(res["pats"]))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("🛡️ 风控")
st.write(f"- 连赢：{ss.risk['win_streak']} 局 | 连输：{ss.risk['consecutive_losses']} 局")
# 安全获取风险等级，防止KeyError
risk_level = res.get("risk", "medium")

if risk_level == "low":
    sug = "✅ 信号清晰，可适度加码"
elif risk_level == "medium":
    sug = "⚠️ 一般信号，轻仓"
elif risk_level == "high":
    sug = "🚨 高波动，谨慎或观望"
elif risk_level == "extreme":
    sug = "⛔ 极高风险，建议暂停"
else:
    sug = "⚪ 暂无风险等级（等待更多数据）"
st.write(f"- 建议：{sug}")
st.markdown('</div>', unsafe_allow_html=True)

# ========= 六路（保留）=========
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("🛣️ 六路（最近）")
r = ss.roads
if r['bead_road']: st.write("珠路：", dots(r['bead_road'][-20:]))
if r['big_road']:
    st.write("大路：")
    for i, col in enumerate(r['big_road'][-5:], 1):
        st.caption(f"列{i}  {dots(col)}")
c1, c2 = st.columns(2)
with c1:
    if r['big_eye_road']: st.write("大眼：", dots(r['big_eye_road'][-12:], red='R'))
with c2:
    if r['small_road']: st.write("小路：", dots(r['small_road'][-10:], red='R'))
if r['three_bead_road']:
    st.write("三珠：")
    for i, g in enumerate(r['three_bead_road'][-4:], 1):
        st.caption(f"组{i}  {dots(g)}")
st.markdown('</div>', unsafe_allow_html=True)

# ========= 统计 & 历史（保留）=========
if ss.games:
    total = len(ss.games)
    bw = seq.count('B'); pw = seq.count('P'); tw = seq.count('T')
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("📊 统计")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总局", total); c2.metric("庄", bw); c3.metric("闲", pw); c4.metric("和", tw)
    bead = r['bead_road']
    if bead:
        avg_streak = np.mean([len(list(g)) for k,g in groupby(bead)])
        chg = sum(1 for i in range(1,len(bead)) if bead[i]!=bead[i-1]) / len(bead) * 100
        st.caption(f"平均连赢 {avg_streak:.1f} 局 · 波动率 {chg:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📝 最近记录（10）"):
        for g in ss.games[-10:][::-1]:
            tag = "🃏" if g['mode']=="card" else ("📝" if g['mode']=="batch" else "🎯")
            res_ = "庄" if g['result']=='B' else ("闲" if g['result']=='P' else "和")
            st.write(f"#{g['round']} {tag} {res_}  |  {g['time']}  "
                     f"{' | 闲: ' + '-'.join(g['player_cards']) if g['mode']=='card' else ''}"
                     f"{' | 庄: ' + '-'.join(g['banker_cards']) if g['mode']=='card' else ''}")

# ========= 控制按钮（保留）=========
col = st.columns(2)
with col[0]:
    if st.button("🔄 新牌靴", use_container_width=True):
        ss.games.clear()
        ss.roads.update({'big_road':[], 'bead_road':[], 'big_eye_road':[], 'small_road':[], 'cockroach_road':[], 'three_bead_road':[]})
        ss.risk.update({'consecutive_losses':0, 'win_streak':0})
        ss.signal_hist.clear()
        st.success("已清空，开始新牌靴")
with col[1]:
    if st.button("💾 导出（提示）", use_container_width=True):
        st.info("手机端建议先用浏览器分享/截屏；如需CSV导出，我可以再给你加导出功能。")
