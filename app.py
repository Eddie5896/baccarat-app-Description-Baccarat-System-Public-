# -*- coding: utf-8 -*-
# Baccarat Master Ultimate - Precision 14.2 Fusion Touch+ 自学习终极版
# 只加不减：保留全部功能 + 单排Touch键 + 和局&对子同行显示

import streamlit as st
import numpy as np
import json
from collections import Counter
from datetime import datetime
from itertools import groupby

st.set_page_config(page_title="🐉 百家乐大师 Precision 14.2 Fusion Touch+", layout="centered")

# ======= 样式 =======
st.markdown("""
<style>
.main-header {font-size:2.2rem;color:#FFD700;text-align:center;text-shadow:2px 2px 4px #000;}
.prediction-card{background:linear-gradient(135deg,#667eea,#764ba2);
    padding:20px;border-radius:15px;border:3px solid #FFD700;margin:15px 0;text-align:center;}
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
.enhanced-logic-panel{background:linear-gradient(135deg,#00b4db,#0083b0);padding:12px;border-radius:10px;margin:10px 0;color:white;}
.touch-row {display:flex; gap:6px; flex-wrap:wrap; align-items:center;}
.touch-btn {padding:8px 10px; font-size:13px; border-radius:8px; border:1px solid #444;
    background:#2c2f36; color:#eee; cursor:pointer;}
.touch-btn-small {padding:6px 8px; font-size:12px;}
.touch-active {background:#FFD54F; color:#111; font-weight:700;}
.card-subline {margin-top:6px; font-size:14px; color:#ffeaa7;}
.small-note {font-size:12px; color:#cbd5e1;}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🐉 百家乐大师 Precision 14.2 Fusion Touch+</h1>', unsafe_allow_html=True)

# ======= 状态初始化 =======
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
    if "prediction_stats" not in ss:
        ss.prediction_stats={'total_predictions':0,'correct_predictions':0,'recent_accuracy':[],'prediction_history':[]}
    if "learning_effectiveness" not in ss: ss.learning_effectiveness=[]
    if "last_prediction" not in ss: ss.last_prediction=None
    # Touch 输入相关
    if "active_side" not in ss: ss.active_side='P'  # 当前输入方：P=闲, B=庄
    if "player_card_text" not in ss: ss.player_card_text=""
    if "banker_card_text" not in ss: ss.banker_card_text=""
    if "input_mode" not in ss: ss.input_mode='card'  # card/result
_init_state()

# ======= 六路分析（保留） =======
class CompleteRoadAnalyzer:
    @staticmethod
    def update_all_roads(result):
        if result not in ['B','P']: return
        roads = st.session_state.expert_roads
        roads['bead_road'].append(result)
        # 大路列
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
        if len(roads['small_路'])>=2: pass  # 兼容老代码（无操作）
        if len(roads['small_road'])>=2:
            ck=[]; r=roads['small_road']
            for i in range(1,len(r)): ck.append('R' if r[i]==r[i-1] else 'B')
            roads['cockroach_road']=ck[-12:]
        if len(roads['bead_road'])>=3:
            br=roads['bead_road']; roads['three_bead_road']=[br[i:i+3] for i in range(0,len(br)-2,3)][-8:]

# ======= 模式检测（保留） =======
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

# ======= 指标核心（保留） =======
class HybridMathCore:
    @staticmethod
    def compute_metrics(seq):
        bp=[x for x in seq if x in ['B','P']]
        if len(bp)<6:
            return {'z':0,'cusum':0,'bayes':0,'momentum':0,'entropy':0,'eor':0}
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

# ======= 自学习核心（保留） =======
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
        hybrid=(m['z']*w['z']+m['cusum']*w['cusum']+m['bayes']*w['bayes']+
                m['momentum']*w['momentum']+m['eor']*w['eor'])
        st.session_state.ai_entropy=m['entropy']
        return hybrid,m

# ======= 状态信号（保留） =======
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
        if roads['big_road'] and roads['big_路']: pass
        if roads['big_road'] and roads['big_road'][-1]:
            if len(roads['big_road'][-1])>=3: sig.append(roads['big_road'][-1][-1])
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

# ======= 风险管理（保留） =======
class ProfessionalRiskManager:
    @staticmethod
    def calculate_position_size(confidence, streak_info):
        base = 1.0
        if confidence > 0.8: base *= 1.2
        elif confidence > 0.7: base *= 1.0
        elif confidence > 0.6: base *= 0.8
        else: base *= 0.5
        if streak_info.get('current_streak', 0) >= 3: base *= 1.1
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
            "low": {"B":"✅ 庄势明确，可适度加仓","P":"✅ 闲势明确，可适度加仓","HOLD":"⚪ 趋势平衡，正常操作"},
            "medium":{"B":"⚠️ 庄势一般，建议轻仓","P":"⚠️ 闲势一般，建议轻仓","HOLD":"⚪ 信号不明，建议观望"},
            "high":{"B":"🚨 高波动庄势，谨慎操作","P":"🚨 高波动闲势，谨慎操作","HOLD":"⛔ 高风险期，建议休息"},
            "extreme":{"B":"⛔ 极高风险，强烈建议观望","P":"⛔ 极高风险，强烈建议观望","HOLD":"⛔ 市场混乱，暂停操作"}
        }
        return suggestions[risk_level].get(direction,"正常操作")

# ======= 概率估计：和局 & 对子（新增显示，不参与主方向） =======
def estimate_tie_pair_probs(metrics, roads):
    """
    简洁可解释的启发式估计：
    - 和局(Tie)基础概率 ~ 9.0%（接近真实区间）
      熵高(混沌)略上调，趋势强(z,cusum大)略下调
    - 对子(任一对)基础：单边对 ~ 7.4%，任一对 = 1-(1-0.074)^2 ≈ 14.2%
      趋势稳定（长龙/动量大）略上调，震荡(单跳)略下调
    """
    ent = float(metrics.get('entropy',0.0))
    z = float(metrics.get('z',0.0))
    cs = float(metrics.get('cusum',0.0))
    mom = float(metrics.get('momentum',0.0))

    # Tie
    tie_base = 0.090
    tie = tie_base + 0.015*min(1.0, ent/1.0) - 0.010*min(1.0, (abs(z)+abs(cs))/2.0)
    tie = float(np.clip(tie, 0.05, 0.14))

    # Pair (either side)
    single_pair = 0.074
    either = 1 - (1-single_pair)**2  # ≈ 0.142
    # 动量大（顺势）+0.01，强震荡（|mom|小 & z小）-0.005
    adj = 0.010*min(1.0, abs(mom)*1.5) - 0.005*(1.0 - min(1.0, abs(z)))
    pair = float(np.clip(either + adj, 0.10, 0.20))
    return tie, pair

# ======= 记录预测/学习（保留） =======
def record_prediction_result(prediction, actual_result, confidence):
    if actual_result in ['B','P']:
        stats = st.session_state.prediction_stats
        stats['total_predictions'] += 1
        is_correct = (prediction == actual_result)
        if is_correct: stats['correct_predictions'] += 1
        stats['recent_accuracy'].append(is_correct)
        if len(stats['recent_accuracy']) > 50: stats['recent_accuracy'].pop(0)
        stats['prediction_history'].append({
            'prediction': prediction,'actual': actual_result,'correct': is_correct,
            'confidence': confidence,'timestamp': datetime.now()
        })

def enhanced_learning_update(prediction, actual_result):
    if prediction in ['B','P'] and actual_result in ['B','P']:
        is_correct = (prediction == actual_result)
        AIHybridLearner.learn_update(correct=is_correct)
        st.session_state.learning_effectiveness.append({
            'correct': is_correct,'weights_snapshot': dict(st.session_state.ai_weights),
            'timestamp': datetime.now()
        })

# ======= 看路推荐（保留） =======
def road_recommendation(roads):
    lines=[]; final=""
    if roads['big_road']:
        last=roads['big_road'][-1]; color_cn="庄" if last[-1]=='B' else "闲"; streak=len(last)
        if streak>=3: lines.append(f"大路：{color_cn}连{streak}局 → 顺路{color_cn}"); final=f"顺大路{color_cn}"
        else: lines.append(f"大路：{color_cn}走势平衡")
    if roads['big_eye_road']:
        r=roads['big_eye_路'].count('R') if 'big_eye_路' in roads else 0
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

# ======= 输入工具 =======
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
    if result in ['B','P']: risk['win_streak']+=1; risk['consecutive_losses']=0
    else: risk['consecutive_losses']+=1; risk['win_streak']=0
    st.success(f"✅ 记录成功! 第{game['round']}局"); st.rerun()

# ======= Touch 输入：单排 A~K + 左右方选择 =======
KEYS = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']

def append_to_active(key_label):
    ss=st.session_state
    if ss.active_side=='P':
        ss.player_card_text = (ss.player_card_text + key_label).upper()
    else:
        ss.banker_card_text = (ss.banker_card_text + key_label).upper()

def backspace_active():
    ss=st.session_state
    if ss.active_side=='P':
        ss.player_card_text = ss.player_card_text[:-1]
    else:
        ss.banker_card_text = ss.banker_card_text[:-1]

def clear_active():
    ss=st.session_state
    if ss.active_side=='P':
        ss.player_card_text = ""
    else:
        ss.banker_card_text = ""

def build_touch_ui():
    ss=st.session_state
    cL, cR = st.columns(2)
    with cL:
        if st.button("🔵 选择闲 (P)", use_container_width=True,
                     type="primary" if ss.active_side=='P' else "secondary"):
            ss.active_side='P'
    with cR:
        if st.button("🔴 选择庄 (B)", use_container_width=True,
                     type="primary" if ss.active_side=='B' else "secondary"):
            ss.active_side='B'

    # 文本框（与 SessionState 同步，不产生冲突）
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("闲家牌", key="player_card_text", value=st.session_state.player_card_text)
    with col2:
        st.text_input("庄家牌", key="banker_card_text", value=st.session_state.banker_card_text)

    # 单排触控键（小按钮，单手好按）
    st.write("")  # 间距
    st.markdown('<div class="touch-row">', unsafe_allow_html=True)
    cols = st.columns(len(KEYS)+3)
    for i,lab in enumerate(KEYS):
        if cols[i].button(lab, key=f"tk_{lab}", help="轻触添加",
                          use_container_width=True):
            append_to_active(lab)
    # 额外：退格、清除、空格(不需要空格，这里省略)
    if cols[-3].button("⌫", key="tk_back", help="退格", use_container_width=True):
        backspace_active()
    if cols[-2].button("清", key="tk_clear", help="清除当前方", use_container_width=True):
        clear_active()
    if cols[-1].button("↩︎ 入", key="tk_commit", help="快捷提交最近一局(需先点结果)", use_container_width=True):
        pass
    st.markdown('</div>', unsafe_allow_html=True)

# ======= 界面：输入系统（保留+Touch增强） =======
def show_quick_start_guide():
    if len(st.session_state.ultimate_games)==0:
        st.markdown("""
        <div class="guide-panel">
        <h3>🎯 快速开始指南</h3>
        <p>1) 选择「牌点输入」或使用「快速看路」</p>
        <p>2) 记录3局后激活AI分析</p>
        <p>3) 左右切换输入方，单排触控键添加 A~K/10</p>
        <p>4) EOR 副数侧栏可调，系统持续学习</p>
        </div>
        """, unsafe_allow_html=True)

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
    show_quick_start_guide()

    c1,c2=st.columns(2)
    with c1:
        if st.button("🃏 牌点输入", use_container_width=True, type="primary"):
            st.session_state.input_mode='card'; st.rerun()
    with c2:
        if st.button("🎯 快速看路", use_container_width=True):
            st.session_state.input_mode='result'; st.rerun()

    if st.session_state.input_mode=='card':
        # Touch 输入 + 文本框（保留键盘输入）
        build_touch_ui()
        st.markdown("### 🏆 本局结果")
        b1,b2,b3=st.columns(3)
        with b1: banker_btn=st.button("🔴 庄赢", use_container_width=True, type="primary")
        with b2: player_btn=st.button("🔵 闲赢", use_container_width=True)
        with b3: tie_btn=st.button("⚪ 和局", use_container_width=True)
        if banker_btn or player_btn or tie_btn:
            handle_card_input(st.session_state.player_card_text,
                              st.session_state.banker_card_text,
                              banker_btn, player_btn, tie_btn)
    else:
        st.info("💡 快速模式：直接记录结果，用于快速看路分析（保留）")
        q1,q2=st.columns(2)
        with q1: qb=st.button("🔴 庄赢", use_container_width=True, type="primary")
        with q2: qp=st.button("🔵 闲赢", use_container_width=True)
        st.markdown("### 📝 批量输入")
        batch=st.text_input("输入BP序列", placeholder="BPBBP 或 庄闲庄庄闲", key="batch_input")
        if st.button("✅ 确认批量输入", use_container_width=True) and batch:
            handle_batch_input(batch)
        if qb or qp: handle_quick_input(qb,qp)

# ======= 智能分析（保留 + 动态阈值 + 投票兜底 + 新增Tie/Pair显示） =======
def display_complete_analysis():
    if len(st.session_state.ultimate_games)<3:
        st.info("🎲 请先记录至少3局牌局数据"); return

    seq=[g['result'] for g in st.session_state.ultimate_games]
    hybrid, metrics = AIHybridLearner.compute_hybrid(seq)

    with st.sidebar:
        decks = st.slider("EOR 计算副数（1-8）", 1, 8, int(st.session_state.eor_decks), key="eor_slider")
        if decks != st.session_state.eor_decks: st.session_state.eor_decks = decks
        st.markdown("### 🤖 AI 权重（只读）")
        w = st.session_state.ai_weights
        st.write({k: round(v,3) for k,v in w.items()})

    state_signals = GameStateDetector.detect(st.session_state.expert_roads)

    st.markdown('<div class="enhanced-logic-panel">', unsafe_allow_html=True)
    st.markdown("#### 🧠 智能决策引擎")
    ent = float(st.session_state.ai_entropy)
    trend = (abs(st.session_state.ai_last_metrics.get('z',0)) +
             abs(st.session_state.ai_last_metrics.get('cusum',0))) / 2.0
    thr_base = 0.10 + 0.04*ent - 0.06*trend
    threshold = float(np.clip(thr_base, 0.05, 0.12))

    hist = st.session_state.prediction_stats.get('prediction_history', [])
    if len(hist) >= 30:
        hold_ratio = np.mean([1 if h['prediction']=='HOLD' else 0 for h in hist[-30:]])
        if hold_ratio > 0.50: threshold *= 0.80

    m = metrics
    def sgn(x): return 'B' if x>0 else ('P' if x<0 else 'HOLD')
    votes = [sgn(m['z']), sgn(m['cusum']), sgn(m['momentum']), sgn(m['bayes']), sgn(m['eor'])]
    cnt = Counter([v for v in votes if v!='HOLD'])
    vote_dir, vote_num = (None,0) if not cnt else cnt.most_common(1)[0]

    if hybrid>threshold: prelim="B"
    elif hybrid<-threshold: prelim="P"
    else: prelim="HOLD"

    margin = abs(hybrid) - threshold
    if prelim!="HOLD" and margin<0.04 and vote_dir in ['B','P'] and vote_dir!=prelim:
        direction = vote_dir
        vote_override=True
    else:
        direction = prelim
        vote_override=False

    scale = 0.12
    sigm = 1/(1 + np.exp(-abs(hybrid)/scale))
    base_conf = 0.52 + 0.36*sigm

    if state_signals:
        for sig in state_signals:
            if '突破' in sig or '共振' in sig:
                base_conf = min(0.94, base_conf*1.12)
            if '衰竭' in sig and direction!='HOLD':
                direction='HOLD'; base_conf=max(base_conf,0.60)

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("动态阈值", f"{threshold:.3f}")
    with c2: st.metric("熵值", f"{ent:.3f}")
    with c3: st.metric("趋势强度", f"{trend:.3f}")
    if vote_override: st.info(f"🎯 投票机制激活: {vote_dir} ({vote_num}/5)")

    st.markdown('</div>', unsafe_allow_html=True)

    # ===== 新增：和局 & 对子概率（同行显示） =====
    tie_p, pair_p = estimate_tie_pair_probs(metrics, st.session_state.expert_roads)

    # 预测卡片
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
        <div class="card-subline">🎲 和局概率：{tie_p*100:.1f}%  |  对子触发率：{pair_p*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # 指标表
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

    # 状态信号展示
    state_signals = GameStateDetector.detect(st.session_state.expert_roads)
    if state_signals:
        for s in state_signals:
            st.markdown(f'<div class="state-signal">🚀 状态信号：{s}</div>', unsafe_allow_html=True)

    # 风险控制
    st.markdown("### 🛡️ 风险控制")
    pos = ProfessionalRiskManager.calculate_position_size(base_conf, {'current_streak':0})
    sug = ProfessionalRiskManager.get_trading_suggestion(risk_level, direction)
    st.markdown(f"""
    <div class="risk-panel">
        <h4 style="color:#fff;margin:0 0 10px 0;">📊 风险控制建议</h4>
        <p style="color:#ccc;margin:5px 0;"><strong>仓位建议:</strong> {pos:.1f} 倍基础仓位</p>
        <p style="color:#ccc;margin:5px 0;"><strong>操作建议:</strong> {sug}</p>
        <p class="small-note">提示：和局/对子概率仅用于信息提示，不参与主方向计算。</p>
    </div>
    """, unsafe_allow_html=True)

    # 学习触发（保持原逻辑示例）
    if len(seq)>0 and direction!='HOLD':
        last_result = seq[-1]
        record_prediction_result(direction, last_result, base_conf)
        enhanced_learning_update(direction, last_result)
        st.session_state.last_prediction = direction

# ======= 六路 / 统计 / 历史（保留） =======
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
    st.markdown("## 📝 完整历史（最近10局）")
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

# ======= 导出（保留） =======
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

# ======= 主程序 =======
def main():
    with st.sidebar:
        st.markdown("## ⚙️ 控制台")
        st.caption("EOR 副数可调；AI 权重后台自学习（只读显示）。")
        st.metric("当前EOR副数", st.session_state.eor_decks)
        # 系统状态简报
        total_games = len(st.session_state.ultimate_games)
        if total_games > 500:
            st.warning("⚠️ 数据量较大，建议导出数据")
        elif total_games > 200:
            st.info("💾 数据量适中，运行流畅")
        else:
            st.success("✅ 系统运行正常")

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
            st.session_state.player_card_text=""
            st.session_state.banker_card_text=""
            st.success("新牌靴开始！"); st.rerun()
    with c2:
        if st.button("📋 导出数据", use_container_width=True):
            enhanced_export_data()

if __name__ == "__main__":
    main()
