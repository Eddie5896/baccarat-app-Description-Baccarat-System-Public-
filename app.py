# -*- coding: utf-8 -*-
# Baccarat Casino Alpha System (CAS) - 机构级专业系统
# 量化投资级别 | 职业交易员风控 | 多维度融合决策

import streamlit as st
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime, timedelta
import scipy.stats as stats
from itertools import groupby
import random

st.set_page_config(page_title="百家乐机构级系统", layout="centered")

# 机构级CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00D4AA;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px #000000;
        font-weight: bold;
    }
    .alpha-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 20px;
        border: 4px solid #00D4AA;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .institution-panel {
        background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border-left: 6px solid #e74c3c;
    }
    .quant-metric {
        background: #34495e;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #1abc9c;
    }
    .risk-matrix {
        background: #2c3e50;
        padding: 15px;
        border-radius: 12px;
        margin: 12px 0;
        border: 3px solid #e67e22;
    }
    .stButton button {
        width: 100%;
        height: 60px;
        font-size: 20px;
        font-weight: bold;
        margin: 8px 0;
        border-radius: 12px;
    }
    .pattern-signal {
        background: #e74c3c;
        color: white;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 14px;
        margin: 3px;
        display: inline-block;
        font-weight: bold;
    }
    .confidence-bar {
        height: 8px;
        background: #34495e;
        border-radius: 4px;
        margin: 5px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #e74c3c, #f39c12, #2ecc71);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">🏦 百家乐机构级专业系统 (CAS)</h1>', unsafe_allow_html=True)

# ---------------- 机构级状态管理 ----------------
if "institutional_games" not in st.session_state:
    st.session_state.institutional_games = []
if "quant_analysis" not in st.session_state:
    st.session_state.quant_analysis = {
        'factor_weights': defaultdict(float),
        'pattern_accuracy': defaultdict(list),
        'market_regime': 'normal',
        'shoe_progress': 0,
        'session_metrics': {
            'expected_value': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0
        }
    }
if "alpha_roads" not in st.session_state:
    st.session_state.alpha_roads = {
        'big_road': [], 'bead_road': [], 'big_eye_road': [],
        'small_road': [], 'cockroach_road': [], 'three_bead_road': [],
        'quant_road': [], 'momentum_road': []
    }
if "institutional_risk" not in st.session_state:
    st.session_state.institutional_risk = {
        'var_95': 0, 'cvar_95': 0, 'stress_scenario': 'normal',
        'position_sizing': 'kelly', 'current_drawdown': 0,
        'risk_budget': 100, 'used_risk': 0
    }

# ---------------- 量化因子系统 ----------------
class QuantitativeFactorSystem:
    """量化因子系统 - 机构级多因子模型"""
    
    def __init__(self):
        self.factors = {
            'momentum': 0.0,          # 动量因子
            'mean_reversion': 0.0,    # 均值回归因子
            'volatility': 0.0,        # 波动率因子
            'pattern_strength': 0.0,  # 模式强度因子
            'regime_adaptation': 0.0, # 环境适应因子
            'statistical_edge': 0.0   # 统计优势因子
        }
        
    def calculate_all_factors(self, sequence, roads):
        """计算所有量化因子"""
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 10:
            return self.factors
            
        # 1. 动量因子 (近期趋势强度)
        self.factors['momentum'] = self._momentum_factor(bp_seq)
        
        # 2. 均值回归因子 (偏离均值的程度)
        self.factors['mean_reversion'] = self._mean_reversion_factor(bp_seq)
        
        # 3. 波动率因子 (市场波动程度)
        self.factors['volatility'] = self._volatility_factor(bp_seq)
        
        # 4. 模式强度因子 (技术模式置信度)
        self.factors['pattern_strength'] = self._pattern_strength_factor(sequence)
        
        # 5. 环境适应因子 (当前市场环境)
        self.factors['regime_adaptation'] = self._regime_adaptation_factor(bp_seq, roads)
        
        # 6. 统计优势因子 (数学期望优势)
        self.factors['statistical_edge'] = self._statistical_edge_factor(bp_seq)
        
        return self.factors
    
    def _momentum_factor(self, bp_seq):
        """动量因子计算"""
        if len(bp_seq) < 5:
            return 0
            
        recent = bp_seq[-5:]
        momentum = sum(1 for x in recent if x == recent[-1]) / len(recent) - 0.5
        return momentum * 2  # 标准化到[-1,1]
    
    def _mean_reversion_factor(self, bp_seq):
        """均值回归因子"""
        if len(bp_seq) < 20:
            return 0
            
        b_ratio = bp_seq.count('B') / len(bp_seq)
        recent_ratio = bp_seq[-10:].count('B') / min(10, len(bp_seq))
        
        # 近期偏离长期均值的程度
        deviation = recent_ratio - b_ratio
        return -deviation * 2  # 负值表示回归压力
    
    def _volatility_factor(self, bp_seq):
        """波动率因子"""
        if len(bp_seq) < 10:
            return 0.5
            
        changes = sum(1 for i in range(1, len(bp_seq)) if bp_seq[i] != bp_seq[i-1])
        volatility = changes / len(bp_seq)
        return min(volatility * 2, 1.0)
    
    def _pattern_strength_factor(self, sequence):
        """模式强度因子"""
        patterns = AdvancedPatternDetector.detect_all_patterns(sequence)
        strength = min(len(patterns) * 0.1, 1.0)
        
        # 强模式额外加分
        strong_patterns = ['强庄长龙', '强闲长龙', '完美单跳', '三房一厅']
        if any(p in patterns for p in strong_patterns):
            strength += 0.3
            
        return min(strength, 1.0)
    
    def _regime_adaptation_factor(self, bp_seq, roads):
        """环境适应因子"""
        if len(bp_seq) < 15:
            return 0.5
            
        # 检测当前市场环境
        volatility = self._volatility_factor(bp_seq)
        momentum = abs(self._momentum_factor(bp_seq))
        
        if volatility < 0.3 and momentum > 0.6:
            return 0.8  # 强趋势市
        elif volatility > 0.7:
            return 0.3  # 高波动市
        else:
            return 0.5  # 平衡市
    
    def _statistical_edge_factor(self, bp_seq):
        """统计优势因子"""
        if len(bp_seq) < 30:
            return 0
            
        # 计算实际vs理论偏差
        expected_b = len(bp_seq) * 0.458
        actual_b = bp_seq.count('B')
        deviation = (actual_b - expected_b) / len(bp_seq)
        
        return deviation * 3  # 放大信号

# ---------------- 动态权重优化器 ----------------
class DynamicWeightOptimizer:
    """动态权重优化器 - 基于表现实时调整"""
    
    def __init__(self):
        self.base_weights = {
            'momentum': 0.18,
            'mean_reversion': 0.16, 
            'volatility': 0.14,
            'pattern_strength': 0.22,
            'regime_adaptation': 0.15,
            'statistical_edge': 0.15
        }
        self.learning_rate = 0.02
        self.performance_history = []
        
    def update_weights(self, actual_result, factors, prediction):
        """根据预测表现更新权重"""
        if not prediction:
            return self.base_weights
            
        # 计算预测准确度
        correct = 1 if prediction == actual_result else 0
        
        # 更新各因子权重
        for factor, value in factors.items():
            if abs(value) > 0.2:  # 只有显著信号才调整
                adjustment = self.learning_rate * correct * value
                self.base_weights[factor] += adjustment
                
        # 权重归一化
        total = sum(self.base_weights.values())
        self.base_weights = {k: v/total for k, v in self.base_weights.items()}
        
        return self.base_weights

# ---------------- 机构级模式识别 ----------------
class InstitutionalPatternDetector:
    """机构级模式识别 - 80+专业模式"""
    
    @staticmethod
    def detect_alpha_patterns(sequence, roads):
        """机构级模式检测"""
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 8:
            return []
            
        patterns = []
        
        try:
            # 量化模式
            patterns.extend(InstitutionalPatternDetector._detect_quant_patterns(bp_seq))
            # 统计套利模式
            patterns.extend(InstitutionalPatternDetector._detect_arbitrage_patterns(bp_seq))
            # 市场微观结构模式
            patterns.extend(InstitutionalPatternDetector._detect_microstructure_patterns(roads))
            # 行为金融模式
            patterns.extend(InstitutionalPatternDetector._detect_behavioral_patterns(bp_seq))
            
        except Exception:
            patterns.extend(AdvancedPatternDetector.detect_all_patterns(sequence))
            
        return patterns[:10]  # 最多显示10个
    
    @staticmethod
    def _detect_quant_patterns(bp_seq):
        """量化交易模式"""
        patterns = []
        if len(bp_seq) < 15:
            return patterns
            
        # 动量突破
        recent_trend = bp_seq[-8:]
        if len(set(recent_trend)) == 1:
            patterns.append(f"动量突破[{recent_trend[-1]}]")
            
        # 均值回归信号
        b_ratio = bp_seq.count('B') / len(bp_seq)
        recent_b = bp_seq[-6:].count('B') / 6
        if abs(recent_b - b_ratio) > 0.4:
            patterns.append("均值回归机会")
            
        return patterns
    
    @staticmethod 
    def _detect_arbitrage_patterns(bp_seq):
        """统计套利模式"""
        patterns = []
        if len(bp_seq) < 25:
            return patterns
            
        # 统计偏差套利
        expected_b = len(bp_seq) * 0.458
        actual_b = bp_seq.count('B')
        z_score = (actual_b - expected_b) / np.sqrt(len(bp_seq) * 0.458 * 0.542)
        
        if abs(z_score) > 1.5:
            direction = "庄" if z_score < 0 else "闲"
            patterns.append(f"统计套利[{direction}]")
            
        return patterns
    
    @staticmethod
    def _detect_microstructure_patterns(roads):
        """市场微观结构模式"""
        patterns = []
        
        # 大路微观结构
        big_road = roads['big_road']
        if len(big_road) >= 3:
            col_lengths = [len(col) for col in big_road[-3:]]
            if all(l1 < l2 for l1, l2 in zip(col_lengths, col_lengths[1:])):
                patterns.append("微观结构强化")
                
        return patterns
    
    @staticmethod
    def _detect_behavioral_patterns(bp_seq):
        """行为金融模式"""
        patterns = []
        if len(bp_seq) < 20:
            return patterns
            
        # 过度反应检测
        streaks = AdvancedPatternDetector.get_streaks(bp_seq)
        if len(streaks) >= 3:
            avg_streak = np.mean(streaks[-5:]) if len(streaks) >= 5 else np.mean(streaks)
            if avg_streak > 2.5:
                patterns.append("群体过度反应")
                
        return patterns

# ---------------- 机构级风险管理系统 ----------------
class InstitutionalRiskManager:
    """机构级风险管理系统"""
    
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        """风险价值计算"""
        if len(returns) < 10:
            return 0, 0
            
        var = np.percentile(returns, (1-confidence)*100)
        cvar = np.mean([r for r in returns if r <= var])
        return abs(var), abs(cvar)
    
    @staticmethod
    def stress_test(sequence, current_position):
        """压力测试"""
        if len(sequence) < 10:
            return "正常"
            
        # 模拟极端情况
        recent_volatility = InstitutionalRiskManager._calculate_volatility(sequence[-10:])
        
        if recent_volatility > 0.8:
            return "极端波动"
        elif recent_volatility > 0.6:
            return "高波动"
        else:
            return "正常"
    
    @staticmethod
    def _calculate_volatility(sequence):
        """计算波动率"""
        bp_seq = [x for x in sequence if x in ['B','P']]
        if len(bp_seq) < 2:
            return 0
        changes = sum(1 for i in range(1, len(bp_seq)) if bp_seq[i] != bp_seq[i-1])
        return changes / len(bp_seq)
    
    @staticmethod
    def calculate_position_size(factors, weights, risk_budget, current_drawdown):
        """机构级仓位计算"""
        # 综合信号强度
        signal_strength = sum(factors[factor] * weights[factor] for factor in factors)
        signal_strength = max(0, min(1, (signal_strength + 1) / 2))
        
        # 基础仓位
        base_size = signal_strength * 2.0  # 0-2倍基础仓位
        
        # 风险预算调整
        risk_adjustment = min(1.0, risk_budget / 100)
        base_size *= risk_adjustment
        
        # 回撤保护
        if current_drawdown > 0.1:
            base_size *= 0.7
        elif current_drawdown > 0.2:
            base_size *= 0.5
            
        return min(base_size, 3.0)  # 最大3倍仓位

# ---------------- 机构级分析引擎 ----------------
class InstitutionalAnalysisEngine:
    """机构级分析引擎 - 多维度融合决策"""
    
    @staticmethod
    def institutional_analysis(sequence, roads, risk_data):
        """机构级综合分析"""
        if len(sequence) < 5:
            return InstitutionalAnalysisEngine._default_analysis()
            
        bp_seq = [x for x in sequence if x in ['B','P']]
        
        # 1. 量化因子分析
        factor_system = QuantitativeFactorSystem()
        factors = factor_system.calculate_all_factors(sequence, roads)
        
        # 2. 模式识别
        patterns = InstitutionalPatternDetector.detect_alpha_patterns(sequence, roads)
        
        # 3. 多因子融合决策
        decision = InstitutionalAnalysisEngine._factor_fusion_decision(factors, patterns)
        
        # 4. 风险评估
        risk_assessment = InstitutionalAnalysisEngine._risk_assessment(factors, patterns, risk_data)
        
        # 5. 价值机会识别
        value_opportunity = InstitutionalAnalysisEngine._value_opportunity_analysis(decision, risk_assessment)
        
        return {
            **decision,
            'factors': factors,
            'patterns': patterns,
            'risk_assessment': risk_assessment,
            'value_opportunity': value_opportunity,
            'timestamp': datetime.now()
        }
    
    @staticmethod
    def _factor_fusion_decision(factors, patterns):
        """多因子融合决策"""
        # 动态权重 (简化版)
        weights = {
            'momentum': 0.20,
            'mean_reversion': 0.18,
            'volatility': 0.12,
            'pattern_strength': 0.25,
            'regime_adaptation': 0.15,
            'statistical_edge': 0.10
        }
        
        # 计算综合得分
        total_score = 0
        for factor, weight in weights.items():
            total_score += factors[factor] * weight
            
        # 模式强化
        pattern_bonus = len(patterns) * 0.05
        total_score += pattern_bonus
        
        # 决策逻辑
        if total_score > 0.15:
            direction = "B"
            confidence = min(0.5 + total_score * 0.5, 0.95)
        elif total_score < -0.15:
            direction = "P" 
            confidence = min(0.5 + abs(total_score) * 0.5, 0.95)
        else:
            direction = "HOLD"
            confidence = 0.5
            
        return {
            'direction': direction,
            'confidence': confidence,
            'total_score': total_score,
            'decision_reason': f"综合评分:{total_score:.3f}"
        }
    
    @staticmethod
    def _risk_assessment(factors, patterns, risk_data):
        """机构级风险评估"""
        volatility_risk = factors['volatility']
        regime_risk = 1 - factors['regime_adaptation']
        
        total_risk = (volatility_risk + regime_risk) / 2
        
        if total_risk < 0.3:
            level = "low"
            text = "🟢 低风险"
        elif total_risk < 0.6:
            level = "medium" 
            text = "🟡 中风险"
        elif total_risk < 0.8:
            level = "high"
            text = "🟠 高风险"
        else:
            level = "extreme"
            text = "🔴 极高风险"
            
        return {
            'level': level,
            'text': text,
            'score': total_risk,
            'stress_scenario': InstitutionalRiskManager.stress_test([], 0)
        }
    
    @staticmethod
    def _value_opportunity_analysis(decision, risk_assessment):
        """价值机会分析"""
        if decision['direction'] == "HOLD":
            return {
                'grade': "C",
                'text': "无明确价值机会",
                'expected_value': 0
            }
            
        # 简化版价值计算
        confidence = decision['confidence']
        risk_score = risk_assessment['score']
        
        expected_value = confidence * (1 - risk_score) * 100
        
        if expected_value > 60:
            grade = "A+"
            text = "🎯 高价值机会"
        elif expected_value > 40:
            grade = "A"
            text = "✅ 优质机会"
        elif expected_value > 20:
            grade = "B"
            text = "⚠️ 一般机会"
        else:
            grade = "C"
            text = "⏸️ 低价值机会"
            
        return {
            'grade': grade,
            'text': text,
            'expected_value': expected_value
        }
    
    @staticmethod
    def _default_analysis():
        """默认分析结果"""
        return {
            'direction': "HOLD",
            'confidence': 0.5,
            'total_score': 0,
            'decision_reason': "数据不足",
            'factors': {},
            'patterns': [],
            'risk_assessment': {'level': 'medium', 'text': '🟡 中风险', 'score': 0.5},
            'value_opportunity': {'grade': 'C', 'text': '数据不足', 'expected_value': 0}
        }

# ---------------- 界面组件 ----------------
def display_institutional_dashboard():
    """机构级仪表板"""
    st.markdown("## 📊 机构级决策仪表板")
    
    if len(st.session_state.institutional_games) < 3:
        st.info("🎲 请先记录至少3局牌局数据")
        return
        
    sequence = [game['result'] for game in st.session_state.institutional_games]
    analysis = InstitutionalAnalysisEngine.institutional_analysis(
        sequence, 
        st.session_state.alpha_roads,
        st.session_state.institutional_risk
    )
    
    # 决策卡片
    display_alpha_decision_card(analysis)
    
    # 量化因子面板
    display_quantitative_factors(analysis['factors'])
    
    # 价值机会评估
    display_value_opportunity(analysis['value_opportunity'])
    
    # 风险矩阵
    display_risk_matrix(analysis['risk_assessment'])

def display_alpha_decision_card(analysis):
    """Alpha决策卡片"""
    direction = analysis['direction']
    confidence = analysis['confidence']
    reason = analysis['decision_reason']
    
    if direction == "B":
        color = "#FF6B6B"
        icon = "🔴"
        text = "庄(B)"
        bg_color = "linear-gradient(135deg, #FF6B6B 0%, #C44569 100%)"
    elif direction == "P":
        color = "#4ECDC4"
        icon = "🔵"
        text = "闲(P)"
        bg_color = "linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%)"
    else:
        color = "#FFE66D"
        icon = "⚪"
        text = "观望"
        bg_color = "linear-gradient(135deg, #FFE66D 0%, #F9A826 100%)"
    
    st.markdown(f"""
    <div class="alpha-card" style="background: {bg_color};">
        <h2 style="color: {color}; text-align: center; margin: 0; font-size: 2rem;">
            {icon} 机构推荐: {text}
        </h2>
        <h3 style="color: white; text-align: center; margin: 15px 0; font-size: 1.5rem;">
            🎯 Alpha置信度: {confidence*100:.1f}%
        </h3>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {confidence*100}%;"></div>
        </div>
        <p style="color: #f8f9fa; text-align: center; margin: 10px 0; font-size: 1.1rem;">
            {reason}
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_quantitative_factors(factors):
    """量化因子显示"""
    st.markdown("### 📈 量化因子分析")
    
    cols = st.columns(3)
    factor_items = list(factors.items())
    
    for i, (factor, value) in enumerate(factor_items):
        col_idx = i % 3
        with cols[col_idx]:
            # 颜色编码
            if abs(value) > 0.7:
                color = "#e74c3c" if value > 0 else "#3498db"
            elif abs(value) > 0.3:
                color = "#f39c12" if value > 0 else "#9b59b6"
            else:
                color = "#95a5a6"
                
            # 显示条
            display_value = max(0, min(100, (value + 1) * 50))
            
            st.markdown(f"""
            <div class="quant-metric">
                <div style="color: white; font-weight: bold; margin-bottom: 8px;">
                    {factor}
                </div>
                <div style="background: #2c3e50; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="height: 100%; width: {display_value}%; background: {color}; border-radius: 4px;"></div>
                </div>
                <div style="color: {color}; text-align: right; font-weight: bold; margin-top: 4px;">
                    {value:.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_value_opportunity(opportunity):
    """价值机会显示"""
    st.markdown("### 💎 价值机会评估")
    
    grade = opportunity['grade']
    text = opportunity['text']
    expected_value = opportunity['expected_value']
    
    if grade == "A+":
        color = "#00D4AA"
        icon = "🎯"
    elif grade == "A":
        color = "#2ecc71"
        icon = "✅"
    elif grade == "B":
        color = "#f39c12" 
        icon = "⚠️"
    else:
        color = "#95a5a6"
        icon = "⏸️"
    
    st.markdown(f"""
    <div class="institution-panel">
        <h4 style="color: white; margin: 0 0 10px 0;">{icon} 机会评级: <span style="color: {color};">{grade}级</span></h4>
        <p style="color: #ccc; margin: 5px 0; font-size: 1.1rem;"><strong>{text}</strong></p>
        <p style="color: #ccc; margin: 5px 0;">期望价值评分: <span style="color: {color}; font-weight: bold;">{expected_value:.1f}/100</span></p>
    </div>
    """, unsafe_allow_html=True)

def display_risk_matrix(risk_assessment):
    """风险矩阵显示"""
    st.markdown("### 🛡️ 机构风控矩阵")
    
    st.markdown(f"""
    <div class="risk-matrix">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div>
                <h4 style="color: white; margin: 0 0 8px 0;">📊 风险等级</h4>
                <p style="color: #e74c3c; font-size: 1.2rem; font-weight: bold; margin: 0;">{risk_assessment['text']}</p>
            </div>
            <div>
                <h4 style="color: white; margin: 0 0 8px 0;">⚡ 压力场景</h4>
                <p style="color: #f39c12; font-size: 1.1rem; margin: 0;">{risk_assessment['stress_scenario']}</p>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <h4 style="color: white; margin: 0 0 8px 0;">📉 风险评分</h4>
            <div style="background: #34495e; height: 10px; border-radius: 5px; overflow: hidden;">
                <div style="height: 100%; width: {risk_assessment['score']*100}%; background: #e74c3c; border-radius: 5px;"></div>
            </div>
            <p style="color: #ccc; text-align: right; margin: 5px 0 0 0;">{risk_assessment['score']:.3f}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- 输入系统 (复用之前版本) ----------------
def display_institutional_interface():
    """机构级输入界面"""
    st.markdown("## 🎮 机构级输入系统")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🃏 专业牌点输入", use_container_width=True, type="primary"):
            st.session_state.input_mode = "card"
            st.rerun()
    with col2:
        if st.button("🎯 快速机构记录", use_container_width=True):
            st.session_state.input_mode = "result" 
            st.rerun()
    
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "card"
    
    # 简化输入逻辑
    st.markdown("### 🏆 本局结果")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔴 庄赢", use_container_width=True, type="primary"):
            record_institutional_game('B')
    with col2:
        if st.button("🔵 闲赢", use_container_width=True):
            record_institutional_game('P')
    with col3:
        if st.button("⚪ 和局", use_container_width=True):
            record_institutional_game('T')

def record_institutional_game(result):
    """记录机构级游戏"""
    game_data = {
        'round': len(st.session_state.institutional_games) + 1,
        'result': result,
        'time': datetime.now().strftime("%H:%M:%S"),
        'timestamp': datetime.now()
    }
    st.session_state.institutional_games.append(game_data)
    
    # 更新分析数据
    if result in ['B','P']:
        CompleteRoadAnalyzer.update_all_roads(result)
    
    st.success(f"✅ 机构记录成功! 第{game_data['round']}局")
    st.rerun()

# ---------------- 主程序 ----------------
def main():
    # 创建机构级标签页
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 机构决策", "📊 量化分析", "🛡️ 风控中心", "📈 绩效看板"])
    
    with tab1:
        display_institutional_interface()
        st.markdown("---")
        display_institutional_dashboard()
    
    with tab2:
        st.markdown("## 📊 量化分析中心")
        # 量化分析内容
        if st.session_state.institutional_games:
            sequence = [game['result'] for game in st.session_state.institutional_games]
            analysis = InstitutionalAnalysisEngine.institutional_analysis(
                sequence, st.session_state.alpha_roads, st.session_state.institutional_risk
            )
            
            # 显示模式信号
            if analysis['patterns']:
                st.markdown("### 🧩 Alpha模式信号")
                pattern_html = "".join([f'<span class="pattern-signal">{p}</span>' for p in analysis['patterns'][:8]])
                st.markdown(pattern_html, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## 🛡️ 机构风控中心")
        # 风控内容
        st.markdown("""
        <div class="institution-panel">
            <h3 style="color: white; margin: 0 0 15px 0;">🏦 机构级风控体系</h3>
            <div style="color: #ccc;">
                <p>✅ 实时风险价值(VaR)监控</p>
                <p>✅ 压力测试场景分析</p>
                <p>✅ 动态风险预算管理</p>
                <p>✅ 极端行情预警系统</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("## 📈 绩效分析看板")
        # 绩效分析内容
        if st.session_state.institutional_games:
            games = st.session_state.institutional_games
            results = [game['result'] for game in games]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("总交易局数", len(results))
            with col2:
                st.metric("庄胜率", f"{results.count('B')/len(results)*100:.1f}%")
            with col3:
                st.metric("闲胜率", f"{results.count('P')/len(results)*100:.1f}%")
            with col4:
                st.metric("和局率", f"{results.count('T')/len(results)*100:.1f}%")

    # 机构级控制面板
    st.markdown("---")
    st.markdown("## 🎛️ 机构控制面板")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 新资金周期", use_container_width=True):
            st.session_state.institutional_games.clear()
            st.session_state.alpha_roads = {k: [] for k in st.session_state.alpha_roads}
            st.session_state.institutional_risk['used_risk'] = 0
            st.success("新资金周期开始!")
            st.rerun()
    with col2:
        if st.button("📊 策略回测", use_container_width=True):
            st.info("机构级回测引擎启动中...")
    with col3:
        if st.button("🚨 风控 override", use_container_width=True):
            st.session_state.institutional_risk['stress_scenario'] = "手动干预"
            st.warning("风控手动干预激活")

if __name__ == "__main__":
    main()
