# -*- coding: utf-8 -*-
# Baccarat Master Ultimate - Precision 15 Apex
# 在 Fusion 14 基础上“只加不减”：自适应权重 + 趋势惯性 + 结构一致性S + 模糊决策 + 反馈记忆
# 不含 Backtest Report；UI 基本不变，仅侧边显示只读引擎状态

import streamlit as st
import numpy as np
import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import groupby

st.set_page_config(page_title="🐉 百家乐大师 Precision 15 Apex", layout="centered")

# ====== 样式（保留） ======
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
.engine-panel{background:#0f172a;border:1px solid #334155;border-radius:10px;padding:10px;margin:8px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐉 百家乐大师 Precision 15 Apex</h1>', unsafe_allow_html=True)

# ====== 状态（保留+新增） ======
def _init_state():
    ss = st.session_state
    if "ultimate_games" not in ss: ss.ultimate_games=[]
    if "expert_roads" not in ss:
        ss.expert_roads={'big_road':[],'bead_road':[],'big_eye_road':[],'small_road':[],'cockroach_road':[],'three_bead_road':[]}
    if "risk_data" not in ss:
        ss.risk_data={'current_level':'medium','position_size':1.0,'stop_loss':3,'consecutive_losses':0,'win_streak':0}
    if "ai_weights" not in ss:
        ss.ai_weights={'z':0.25,'cusum':0.25,'bayes':0.20,'momentum':0.15,'eor':0.15}
    if "ai_learning_buffer" not in ss: ss.ai_learning_buffer=[]
    if "ai_last_metrics" not in ss: ss.ai_last_metrics={}
    if "ai_entropy" not in ss: ss.ai_entropy=0.0
    if "eor_decks" not in ss: ss.eor_decks=7
    if "ai_batch_n" not in ss: ss.ai_batch_n=5

    # 预测统计（保留）
    if "prediction_stats" not in ss:
        ss.prediction_stats = {'total_predictions':0,'correct_predictions':0,'recent_accuracy':[],'prediction_history':[]}

    # ===== 新增：引擎与反馈记忆 =====
    if "engine_state" not in ss:
        ss.engine_state = {
            'window': 10,                # 动态权重窗口
            'trend_inertia_on': True,    # 趋势惯性开关
            'structure_S': 0.0,          # 大眼/小路/蟑螂一致性
            'dynamic_threshold': 0.10,   # 动态出手阈值（只读展示）
            'hold_relax': 1.0,           # HOLD 放宽倍数（只读展示）
            'vote_override': False,      # 投票是否触发（只读）
        }
    if "feedback_memory" not in ss:
        ss.feedback_memory = {
            'long_run_wrong': 0,   # 长龙时逆向错
            'chop_wrong': 0,       # 震荡时跟随错
            'recent_run': [],      # 近期正确/错误序列布尔
            'max_consec_wrong': 0, # 统计最大连错
        }

_init_state()

# ====== 六路（保留） ======
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
        # 大眼/小路/蟑螂/三珠（简化兼容原逻辑）
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
            ck=[]; r=roads['small_路'] if 'small_路' in roads else roads['small_road']
            # 兼容性修正
            r = roads['small_road']
            for i in range(1,len(r)): ck.append('R' if r[i]==r[i-1] else 'B')
            roads['cockroach_road']=ck[-12:]
        if len(roads['bead_road'])>=3:
            br=roads['bead_road']; roads['three_bead_road']=[br[i:i+3] for i in range(0,len(br)-2,3)][-8:]

# ====== 模式检测（保留） ======
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

# ====== 指标核心（保留） ======
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

# ====== AI 学习（保留）+ 自适应权重（新增） ======
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
    def adaptive_rebalance():
        """根据最近窗口命中情况，微调各权重（不改UI）"""
        ss=st.session_state
        ps=ss.prediction_stats['prediction_history']
        if len(ps)<ss.engine_state['window']: return
        recent=ps[-ss.engine_state['window']:]
        acc=np.mean([1 if x['correct'] else 0 for x in recent]) if recent else 0.0
        w=ss.ai_weights
        # 根据最近“成功的指标倾向”来放大/缩小（启发式）
        lm=ss.ai_last_metrics or {}
        z,c,b,m,e = abs(lm.get('z',0)), abs(lm.get('cusum',0)), abs(lm.get('bayes',0)), abs(lm.get('momentum',0)), abs(lm.get('eor',0))
        mag = np.array([z,c,b,m,e]); mag = mag/(mag.sum()+1e-6)
        gain = 0.02*(acc-0.5)  # 最近 >50% 则正向增益
        keys=['z','cusum','bayes','momentum','eor']
        for i,k in enumerate(keys):
            w[k]=float(np.clip(w[k]*(1+gain*mag[i]),0.05,0.4))

# ====== 状态信号（保留） ======
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
    def detect(roads):
        out=[]
        # 简洁：大路连续>=3提示
        if roads['big_road'] and len(roads['big_road'][-1])>=3:
            last=roads['big_road'][-1][-1]
            out.append(f"大路突破-{'庄' if last=='B' else '闲'}势增强")
        # 辅路一致性
        sig=[]
        if roads['big_eye_road'][-3:].count('R')==3: sig.append('B')
        if roads['small_road'][-3:].count('R')==3: sig.append('B')
        if roads['big_eye_road'][-3:].count('B')==3: sig.append('P')
        if roads['small_road'][-3:].count('B')==3: sig.append('P')
        if sig:
            mc=Counter(sig).most_common(1)[0]
            if mc[1]>=2: out.append(f"多路共振-{'庄趋势' if mc[0]=='B' else '闲趋势'}")
        # 龙衰竭
        bead=roads['bead_road']
        if bead:
            streak=GameStateDetector._get_current_streak(bead)
            if streak>=5: out.append(f"连势衰竭-{'庄' if bead[-1]=='B' else '闲'}龙衰竭")
        return out

# ====== 结构一致性分数 S（新增） ======
def compute_structure_S(roads):
    """大眼/小路/蟑螂三路一致性：S∈[-1,1]（偏庄为正、偏闲为负）"""
    score=0; cnt=0
    # 转换：R->B倾向, B->P倾向（与传统路法对应）
    for key in ['big_eye_road','small_road','cockroach_road']:
        r = roads.get(key, [])
        if not r: continue
        tail=r[-6:]  # 最近6格
        rb = tail.count('R'); bb = tail.count('B')
        if rb+bb==0: continue
        sc = (rb - bb) / (rb + bb)  # R多→正，B多→负
        score += sc; cnt += 1
    S = (score/cnt) if cnt>0 else 0.0
    st.session_state.engine_state['structure_S']=float(np.clip(S,-1,1))
    return st.session_state.engine_state['structure_S']

# ====== 风险管理（保留） ======
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
            "low": {"B": "✅ 庄势明确，可适度加仓","P": "✅ 闲势明确，可适度加仓","HOLD": "⚪ 趋势平衡，正常操作"},
            "medium": {"B": "⚠️ 庄势一般，建议轻仓","P": "⚠️ 闲势一般，建议轻仓","HOLD": "⚪ 信号不明，建议观望"},
            "high": {"B": "🚨 高波动庄势，谨慎操作","P": "🚨 高波动闲势，谨慎操作","HOLD": "⛔ 高风险期，建议休息"},
            "extreme": {"B": "⛔ 极高风险，强烈建议观望","P": "⛔ 极高风险，强烈建议观望","HOLD": "⛔ 市场混乱，暂停操作"}
        }
        return suggestions[risk_level].get(direction, "正常操作")

# ====== 记录/学习（保留） ======
def record_prediction_result(prediction, actual_result, confidence):
    if actual_result in ['B', 'P']:
        stats = st.session_state.prediction_stats
        stats['total_predictions'] += 1
        is_correct = (prediction == actual_result)
        if is_correct: stats['correct_predictions'] += 1
        stats['recent_accuracy'].append(is_correct)
        if len(stats['recent_accuracy']) > 50: stats['recent_accuracy'].pop(0)
        stats['prediction_history'].append({
            'prediction': prediction,'actual': actual_result,
            'correct': is_correct,'confidence': confidence,'timestamp': datetime.now()
        })
        # 反馈记忆
        fm=st.session_state.feedback_memory
        fm['recent_run'].append(is_correct)
        # 更新最大连错
        consec=0; mx=0
        for ok in fm['recent_run'][::-1]:
            if not ok: consec+=1; mx=max(mx,consec)
            else: break
        fm['max_consec_wrong']=max(fm['max_consec_wrong'], mx)

def enhanced_learning_update(prediction, actual_result):
    if prediction in ['B','P'] and actual_result in ['B','P']:
        is_correct = (prediction == actual_result)
        AIHybridLearner.learn_update(correct=is_correct)

# ====== 看路推荐（保留） ======
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
        last3=roads['cockroach_路'] if 'cockroach_路' in roads else roads['cockroach_road']
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

# ====== 输入/记录（保留） ======
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
    # 风险状态
    risk=st.session_state.risk_data
    if result in ['B','P']: risk['win_streak']+=1; risk['consecutive_losses']=0
    else: risk['consecutive_losses']+=1; risk['win_streak']=0
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
    s=batch_input.upper().replace('庄','B').replace('闲','P').replace(' ','')
    valid=[c for c in s if c in ['B','P']]
    if valid:
        for r in valid: record_game(r,['X','X'],['X','X'],'batch')
        st.success(f"✅ 批量添加{len(valid)}局")

def display_complete_interface():
    st.markdown("## 🎮 双模式输入系统")
    if len(st.session_state.ultimate_games)==0:
        st.markdown("""
        <div class="guide-panel">
        <h3>🎯 快速开始</h3>
        <p>1) 记录 3 局后自动启动智能分析；2) EOR 副数可在侧边栏调节；3) 风险建议仅作参考。</p>
        </div>""", unsafe_allow_html=True)
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
        st.info("💡 快速模式：直接记录结果，用于快速看路分析")
        q1,q2=st.columns(2)
        with q1: qb=st.button("🔴 庄赢", use_container_width=True, type="primary")
        with q2: qp=st.button("🔵 闲赢", use_container_width=True)
        st.markdown("### 📝 批量输入")
        batch=st.text_input("输入BP序列", placeholder="BPBBP 或 庄闲庄庄闲", key="batch_input")
        if st.button("✅ 确认批量输入", use_container_width=True) and batch:
            handle_batch_input(batch)
        if qb or qp: handle_quick_input(qb,qp)

# ====== 智能分析（升级） ======
def display_complete_analysis():
    if len(st.session_state.ultimate_games)<3:
        st.info("🎲 请先记录至少3局牌局数据"); return

    seq=[g['result'] for g in st.session_state.ultimate_games]
    hybrid, metrics = AIHybridLearner.compute_hybrid(seq)

    # 侧边栏：EOR 与权重只读
    with st.sidebar:
        decks = st.slider("EOR 计算副数（1-8）", 1, 8, int(st.session_state.eor_decks), key="eor_slider")
        if decks != st.session_state.eor_decks: st.session_state.eor_decks = decks
        st.markdown("### 🤖 AI 权重（只读）")
        st.write({k: round(v,3) for k,v in st.session_state.ai_weights.items()})

    # ===== 结构一致性 S =====
    S = compute_structure_S(st.session_state.expert_roads)

    # ===== 动态阈值：熵↑ -> 阈值↑；趋势强( |z|/cusum ) -> 阈值↓；再用 S 微调 =====
    ent = float(metrics['entropy']); trend = (abs(metrics['z'])+abs(metrics['cusum']))/2.0
    thr = 0.10 + 0.05*ent - 0.06*trend - 0.03*abs(S)
    threshold = float(np.clip(thr, 0.05, 0.12))
    st.session_state.engine_state['dynamic_threshold']=threshold

    # ===== HOLD 软夹子：近30笔 HOLD 多则放宽阈值 20% =====
    hist = st.session_state.prediction_stats.get('prediction_history', [])
    hold_adjust = 1.0
    if len(hist)>=30:
        hold_ratio = np.mean([1 if h['prediction']=='HOLD' else 0 for h in hist[-30:]])
        if hold_ratio>0.50:
            threshold *= 0.80; hold_adjust=0.80
    st.session_state.engine_state['hold_relax']=hold_adjust

    # ===== 投票兜底 =====
    m=metrics
    def sgn(x): return 'B' if x>0 else ('P' if x<0 else 'HOLD')
    votes=[sgn(m['z']), sgn(m['cusum']), sgn(m['momentum']), sgn(m['bayes']), sgn(m['eor'])]
    cnt=Counter([v for v in votes if v!='HOLD'])
    vote_dir,vote_num=(None,0) if not cnt else cnt.most_common(1)[0]

    # 初判
    if hybrid>threshold: prelim="B"
    elif hybrid<-threshold: prelim="P"
    else: prelim="HOLD"
    margin = abs(hybrid)-threshold
    vote_override=False
    if prelim!="HOLD" and margin<0.04 and vote_dir in ['B','P'] and vote_dir!=prelim:
        direction=vote_dir; vote_override=True
    else:
        direction=prelim
    st.session_state.engine_state['vote_override']=vote_override

    # ===== 趋势惯性：连对/连错的动态管理（只后台，不加滑杆） =====
    fm=st.session_state.feedback_memory
    recent_correct = fm['recent_run']
    # 估算最近连续对/错
    consec_right = 0
    for ok in reversed(recent_correct):
        if ok: consec_right+=1
        else: break
    consec_wrong = 0
    for ok in reversed(recent_correct):
        if not ok: consec_wrong+=1
        else: break

    # 置信度重标定 + 状态信号增强
    scale = 0.12
    sigm = 1/(1 + np.exp(-abs(hybrid)/scale))
    base_conf = 0.52 + 0.36*sigm  # 0.52~0.88

    # 结构 S 推动：|S|越大，按方向加/减 2~4%
    base_conf *= (1 + 0.04*abs(S))

    # 状态信号
    state_signals = GameStateDetector.detect(st.session_state.expert_roads)
    if state_signals:
        for sig in state_signals:
            if '突破' in sig or '共振' in sig:
                base_conf=min(0.95, base_conf*1.10)
            if '衰竭' in sig and direction!='HOLD':
                direction='HOLD'
                base_conf=max(base_conf,0.60)

    # 连续对/错惯性
    if consec_right>=3 and direction!='HOLD':
        base_conf=min(0.97, base_conf*1.06)  # 胜率段加一点“胆量”
    if consec_wrong>=3:
        # 连错则提高门槛（等价于方向变保守）
        threshold=min(0.14, threshold*1.20)
        st.session_state.engine_state['dynamic_threshold']=threshold

    # 风险与卡片展示
    vol = float(abs(metrics['momentum']))*0.6 + 0.4*(1 - abs(metrics['bayes']))
    risk_level, risk_text = ProfessionalRiskManager.get_risk_level(base_conf, vol)

    # 推荐卡片
    if direction=="B":
        color="#FF6B6B"; icon="🔴"; text="庄(B)"; bg="linear-gradient(135deg,#FF6B6B,#C44569)"
    elif direction=="P":
        color="#4ECDC4"; icon="🔵"; text="闲(P)"; bg="linear-gradient(135deg,#4ECDC4,#44A08D)"
    else:
        color="#FFE66D"; icon="⚪"; text="观望"; bg="linear-gradient(135deg,#FFE66D,#F9A826)"

    st.markdown(f"""
    <div class="prediction-card" style="background:{bg};">
        <h2 style="color:{color};margin:0;text-align:center;">{icon} 大师推荐: {'庄(B)' if direction=='B' else ('闲(P)' if direction=='P' else '观望')}</h2>
        <h3 style="color:#fff;text-align:center;margin:10px 0;">🎯 置信度: {base_conf*100:.1f}% | {risk_text}</h3>
        <p style="color:#f8f9fa;text-align:center;margin:0;">
            结构一致性 S: {S:+.2f} | 动态阈值: {threshold:.3f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 指标表
    def badge(v):
        if v>0: return f'<span class="badge badge-pos">+{v:.3f}</span>'
        if v<0: return f'<span class="badge badge-neg">{v:.3f}</span>'
        return f'<span class="badge badge-neutral">{v:.3f}</span>'
    w = st.session_state.ai_weights
    st.markdown(f"""
    <div class="metric-table">
      <div class="row"><div>Z-Score</div><div>{badge(metrics['z'])} · w={w['z']:.2f}</div></div>
      <div class="row"><div>CUSUM</div><div>{badge(metrics['cusum'])} · w={w['cusum']:.2f}</div></div>
      <div class="row"><div>Bayes</div><div>{badge(metrics['bayes'])} · w={w['bayes']:.2f}</div></div>
      <div class="row"><div>Momentum</div><div>{badge(metrics['momentum'])} · w={w['momentum']:.2f}</div></div>
      <div class="row"><div>EOR (decks={st.session_state.eor_decks})</div><div>{badge(metrics['eor'])} · w={w['eor']:.2f}</div></div>
      <div class="row"><div>Entropy</div><div>{badge(metrics['entropy'])}</div></div>
      <div class="row"><div><b>Hybrid 合成</b></div><div><b>{badge(hybrid)}</b></div></div>
      <div class="row"><div>方向</div><div><b>{'庄(B)' if direction=='B' else ('闲(P)' if direction=='P' else '观望')}</b></div></div>
    </div>
    """, unsafe_allow_html=True)

    # 风险面板
    st.markdown("### 🛡️ 风险控制")
    pos = ProfessionalRiskManager.calculate_position_size(base_conf, {'current_streak':0})
    sug = ProfessionalRiskManager.get_trading_suggestion(risk_level, direction)
    st.markdown(f"""
    <div class="risk-panel">
        <h4 style="color:#fff;margin:0 0 10px 0;">📊 风险控制建议</h4>
        <p style="color:#ccc;margin:5px 0;"><strong>仓位建议:</strong> {pos:.1f} 倍基础仓位</p>
        <p style="color:#ccc;margin:5px 0;"><strong>操作建议:</strong> {sug}</p>
        <p style="color:#ccc;margin:5px 0;"><strong>连赢:</strong> {st.session_state.risk_data['win_streak']} 局 | <strong>连输:</strong> {st.session_state.feedback_memory['max_consec_wrong']}（历史最大连错）</p>
    </div>
    """, unsafe_allow_html=True)

    # 写回最后指标
    st.session_state.ai_last_metrics=metrics
    st.session_state.ai_entropy=metrics['entropy']

    # 学习与统计（使用上一手真实结果）
    if len(seq)>0 and direction!='HOLD':
        last_result=seq[-1]
        record_prediction_result(direction, last_result, base_conf)
        enhanced_learning_update(direction, last_result)
        # 自适应权重
        AIHybridLearner.adaptive_rebalance()

# ====== 六路展示 / 统计 / 历史（保留） ======
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
    # 预测性能
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

# ====== 侧边：系统状态（保留+新增只读引擎状态） ======
def add_system_status_panel():
    with st.sidebar.expander("📊 系统状态", expanded=False):
        total_games = len(st.session_state.ultimate_games)
        st.metric("总局数", total_games)
        stats = st.session_state.prediction_stats
        if stats['total_predictions'] > 0:
            accuracy = (stats['correct_predictions'] / stats['total_predictions']) * 100
            st.metric("预测准确率", f"{accuracy:.1f}%")
            st.metric("总预测数", stats['total_predictions'])
        fm = st.session_state.feedback_memory
        st.metric("历史最大连错", fm['max_consec_wrong'])
        eng = st.session_state.engine_state
        st.markdown("### ⚙️ 智能引擎（只读）")
        st.write({
            "window": eng['window'],
            "structure_S": round(eng['structure_S'],3),
            "dynamic_threshold": round(eng['dynamic_threshold'],3),
            "hold_relax": round(eng['hold_relax'],2),
            "vote_override": eng['vote_override']
        })

# ====== 主程序（保留） ======
def main():
    with st.sidebar:
        st.markdown("## ⚙️ 控制台")
        st.caption("EOR 副数可调；AI 自动自适应，界面不增滑杆。")
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
            st.session_state.prediction_stats={'total_predictions':0,'correct_predictions':0,'recent_accuracy':[],'prediction_history':[]}
            st.session_state.engine_state.update({'structure_S':0.0,'dynamic_threshold':0.10,'hold_relax':1.0,'vote_override':False})
            st.session_state.feedback_memory={'long_run_wrong':0,'chop_wrong':0,'recent_run':[],'max_consec_wrong':0}
            st.success("新牌靴开始！"); st.rerun()
    with c2:
        if st.button("📋 导出数据", use_container_width=True):
            data = {
                'games': st.session_state.ultimate_games,
                'roads': st.session_state.expert_roads,
                'ai_weights': st.session_state.ai_weights,
                'prediction_stats': st.session_state.prediction_stats,
                'engine_state': st.session_state.engine_state,
                'feedback_memory': st.session_state.feedback_memory,
                'export_time': datetime.now().isoformat()
            }
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            st.download_button("📥 下载完整数据", json_str,
                file_name=f"baccarat_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json")

if __name__ == "__main__":
    main()
