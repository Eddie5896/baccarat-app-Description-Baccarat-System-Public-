# -*- coding: utf-8 -*-
# Baccarat Master – Mobile Lite（手机稳定版）
# 保留：六路 / 60+模式 / 牌点增强 / 风控面板
# 精简：重样式、频繁 rerun、复杂 HTML

import streamlit as st
import numpy as np
from itertools import groupby
from datetime import datetime

st.set_page_config(page_title="Baccarat Mobile Lite", layout="centered")

# ===== 轻量样式（纯色+紧凑间距，适配手机） =====
st.markdown("""
<style>
  .h1 {font-size: 1.4rem; font-weight:700; text-align:center; margin: .2rem 0 .6rem;}
  .card {background:#1f2937; border:1px solid #374151; border-radius:10px; padding:.8rem; margin:.5rem 0;}
  .pill {display:inline-block; padding:.2rem .5rem; border-radius:999px; font-size:.8rem; margin:.15rem; color:#fff;}
  .pill-r {background:#ef4444;} .pill-b {background:#3b82f6;}
  .pill-g {background:#10b981;} .pill-y {background:#f59e0b;}
  .mono {font-family: ui-monospace, SFMono-Regular, Menlo, monospace;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="h1">🐉 Baccarat Master — Mobile Lite</div>', unsafe_allow_html=True)

# ===== SessionState（手机端避免 KeyError）=====
ss = st.session_state
ss.setdefault("games", [])  # [{'round', 'player_cards', 'banker_cards', 'result', 'time', 'mode'}]
ss.setdefault("roads", {'big_road':[], 'bead_road':[], 'big_eye_road':[], 'small_road':[], 'cockroach_road':[], 'three_bead_road':[]})
ss.setdefault("risk", {'consecutive_losses':0, 'win_streak':0})

# ===== 六路生成（轻量实现）=====
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

# ===== 模式识别（核心保留，写成轻量函数）=====
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

# ===== 牌点增强（轻量版，±20% 置信度微调）=====
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

# ===== 主分析（保持核心逻辑，去掉 heavy UI）=====
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

def analyze(sequence, games):
    if len(sequence) < 3:
        return {"dir":"HOLD","conf":0.5,"pats":[],"reason":"数据不足","vol":0.0,"streak":0,"risk":"medium","risk_text":"🟡 中风险"}
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

    # base conf
    conf = min(0.9, 0.5 + abs(score)*0.4 + len(pats)*0.1)
    # direction
    if score > 0.15: d = "B"
    elif score < -0.15: d = "P"
    else: d="HOLD"; conf=0.5

    # 牌点增强（只微调置信度）
    enh, enh_txt = CardEnh.analyze(games)
    if enh != 0:
        conf = max(0.1, min(0.95, conf + enh))

    # 风险等级（简化）
    risk_score = (1-conf) + vol
    if risk_score < .3: risk=("low","🟢 低风险")
    elif risk_score < .6: risk=("medium","🟡 中风险")
    elif risk_score < .8: risk=("high","🟠 高风险")
    else: risk=("extreme","🔴 极高风险")

    reason_bits=[]
    if pats: reason_bits.append("模式:"+",".join(pats[:3]))
    if s>=2: reason_bits.append(f"连{s}")
    reason_bits.append(f"风险:{risk[0]}")
    if enh_txt: reason_bits.append("牌点:"+enh_txt)
    if d=="HOLD": reason_bits.append("建议观望")

    return {"dir":d,"conf":conf,"pats":pats,"reason":" | ".join(reason_bits),
            "vol":vol,"streak":s,"risk":risk[0],"risk_text":risk[1]}

# ===== 工具：解析牌点输入（手机友好）=====
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

# ===== 顶部：输入（用 form 防抖，避免手机端重复提交）=====
with st.form("input_form"):
    st.write("🎮 录入一局（建议用本表单，手机更稳）")
    c1, c2 = st.columns(2)
    with c1:
        p_in = st.text_input("闲家牌 (例: K10 或 552)", key="p_in")
    with c2:
        b_in = st.text_input("庄家牌 (例: 55 或 AJ)", key="b_in")

    col = st.columns(3)
    with col[0]: choose_b = st.form_submit_button("录入 庄赢", use_container_width=True)
    with col[1]: choose_p = st.form_submit_button("录入 闲赢", use_container_width=True)
    with col[2]: choose_t = st.form_submit_button("录入 和局", use_container_width=True)

# 按钮处理（无 rerun，直接更新内存并显示）
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
        # 风险计数
        if result in ['B','P']:
            ss.risk['win_streak'] += 1
            ss.risk['consecutive_losses'] = 0
        else:
            ss.risk['consecutive_losses'] += 1
            ss.risk['win_streak'] = 0
    st.toast(f"✅ 第 {len(ss.games)} 局已记录（{ '庄' if result=='B' else '闲' if result=='P' else '和' }）")

# 批量输入（手机端一行搞定）
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

# ===== 分析卡片 =====
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("🎯 智能分析")
seq = [g['result'] for g in ss.games]
res = analyze(seq, ss.games) if len(seq)>=1 else {"dir":"HOLD","conf":0.5,"pats":[],"reason":"等待数据","risk_text":"🟡 中风险"}
dir_map = {"B":("庄","pill-r"), "P":("闲","pill-b"), "HOLD":("观望","pill-y")}
name, cls = dir_map.get(res["dir"], ("观望","pill-y"))
st.markdown(f'<span class="pill {cls}">推荐：{name}</span>  '
            f'<span class="pill pill-g">置信度：{res["conf"]*100:.1f}%</span>  '
            f'<span class="pill pill-y">{res["risk_text"]}</span>', unsafe_allow_html=True)
st.caption(res["reason"])
st.markdown('</div>', unsafe_allow_html=True)

# ===== 模式 & 风控 =====
if res.get("pats"):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("🧩 检测模式")
    st.write(", ".join(res["pats"]))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("🛡️ 风控")
st.write(f"- 连赢：{ss.risk['win_streak']} 局 | 连输：{ss.risk['consecutive_losses']} 局")
# 简化仓位建议（手机端易读）
if res["risk"]=="low": sug = "✅ 信号清晰，可适度加码"
elif res["risk"]=="medium": sug = "⚠️ 一般信号，轻仓"
elif res["risk"]=="high": sug = "🚨 高波动，谨慎或观望"
else: sug = "⛔ 极高风险，建议暂停"
st.write(f"- 建议：{sug}")
st.markdown('</div>', unsafe_allow_html=True)

# ===== 六路（紧凑显示，适配手机）=====
def dots(arr, red='B'):
    return " ".join('🔴' if x==red or x=='R' else '🔵' for x in arr)

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

# ===== 统计 & 历史（手机简版）=====
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

# 重置按钮（无 rerun，安全清空）
col = st.columns(2)
with col[0]:
    if st.button("🔄 新牌靴", use_container_width=True):
        ss.games.clear()
        ss.roads.update({'big_road':[], 'bead_road':[], 'big_eye_road':[], 'small_road':[], 'cockroach_road':[], 'three_bead_road':[]})
        ss.risk.update({'consecutive_losses':0, 'win_streak':0})
        st.success("已清空，开始新牌靴")
with col[1]:
    if st.button("💾 导出（提示）", use_container_width=True):
        st.info("手机端建议先用浏览器分享/截屏；需要CSV导出我再给你加。")
