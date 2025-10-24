# 在现有代码的 AdvancedPatternDetector 类之后添加以下新类

# ---------------- 牌点增强分析系统 ----------------
class CardEnhancementAnalyzer:
    """牌点增强分析系统 - 不改变原有逻辑，只做增强"""
    
    @staticmethod
    def analyze_card_enhancement(games_with_cards):
        """分析牌点数据，返回增强系数"""
        if len(games_with_cards) < 3:
            return {"enhancement_factor": 0, "reason": "数据不足"}
            
        # 只分析有具体牌点的游戏
        card_games = [game for game in games_with_cards if game.get('mode') == 'card' 
                     and len(game['player_cards']) >= 2 and len(game['banker_cards']) >= 2]
        
        if len(card_games) < 2:
            return {"enhancement_factor": 0, "reason": "牌点数据不足"}
        
        enhancement = 0
        reasons = []
        
        try:
            # 1. 天牌效应分析
            natural_effect = CardEnhancementAnalyzer._analyze_natural_effect(card_games)
            if natural_effect['factor'] != 0:
                enhancement += natural_effect['factor']
                reasons.append(natural_effect['reason'])
            
            # 2. 点数动量分析
            point_momentum = CardEnhancementAnalyzer._analyze_point_momentum(card_games)
            if point_momentum['factor'] != 0:
                enhancement += point_momentum['factor']
                reasons.append(point_momentum['reason'])
            
            # 3. 补牌模式分析
            draw_pattern = CardEnhancementAnalyzer._analyze_draw_patterns(card_games)
            if draw_pattern['factor'] != 0:
                enhancement += draw_pattern['factor']
                reasons.append(draw_pattern['reason'])
                
        except Exception as e:
            # 如果分析出错，不影响主系统
            return {"enhancement_factor": 0, "reason": "分析异常"}
        
        return {
            "enhancement_factor": max(-0.2, min(0.2, enhancement)),  # 限制在±20%以内
            "reason": " | ".join(reasons) if reasons else "无增强信号"
        }
    
    @staticmethod
    def _analyze_natural_effect(card_games):
        """天牌效应分析"""
        if len(card_games) < 3:
            return {"factor": 0, "reason": ""}
            
        # 检测最近是否有天牌
        recent_games = card_games[-3:]
        natural_count = 0
        
        for game in recent_games:
            player_points = CardEnhancementAnalyzer._calculate_points(game['player_cards'])
            banker_points = CardEnhancementAnalyzer._calculate_points(game['banker_cards'])
            
            if player_points >= 8 or banker_points >= 8:
                natural_count += 1
        
        if natural_count >= 2:
            return {"factor": 0.08, "reason": f"天牌密集({natural_count}局)"}
        elif natural_count == 1:
            return {"factor": 0.03, "reason": "天牌出现"}
        else:
            return {"factor": 0, "reason": ""}
    
    @staticmethod
    def _analyze_point_momentum(card_games):
        """点数动量分析"""
        if len(card_games) < 4:
            return {"factor": 0, "reason": ""}
            
        # 计算近期点数平均值
        recent_points = []
        for game in card_games[-4:]:
            player_points = CardEnhancementAnalyzer._calculate_points(game['player_cards'])
            banker_points = CardEnhancementAnalyzer._calculate_points(game['banker_cards'])
            recent_points.extend([player_points, banker_points])
        
        avg_points = sum(recent_points) / len(recent_points)
        
        if avg_points < 4:
            return {"factor": 0.06, "reason": "小点数期"}
        elif avg_points > 7:
            return {"factor": -0.04, "reason": "大点数期"}
        else:
            return {"factor": 0, "reason": ""}
    
    @staticmethod
    def _analyze_draw_patterns(card_games):
        """补牌模式分析"""
        if len(card_games) < 5:
            return {"factor": 0, "reason": ""}
            
        # 统计需要补牌的概率
        draw_count = 0
        total_analyzed = min(10, len(card_games))
        
        for game in card_games[-total_analyzed:]:
            player_points = CardEnhancementAnalyzer._calculate_points(game['player_cards'])
            banker_points = CardEnhancementAnalyzer._calculate_points(game['banker_cards'])
            
            # 简化版补牌判断
            if player_points < 6 or banker_points < 6:
                draw_count += 1
        
        draw_ratio = draw_count / total_analyzed
        
        if draw_ratio > 0.7:
            return {"factor": -0.05, "reason": "补牌密集"}
        elif draw_ratio < 0.3:
            return {"factor": 0.04, "reason": "补牌稀少"}
        else:
            return {"factor": 0, "reason": ""}
    
    @staticmethod
    def _calculate_points(cards):
        """计算牌点"""
        point_map = {
            'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
            '7': 7, '8': 8, '9': 9, '10': 0, 'J': 0, 'Q': 0, 'K': 0
        }
        
        total = 0
        for card in cards:
            total += point_map.get(card, 0)
        
        return total % 10  # 只取个位数

# ---------------- 增强版分析引擎 ----------------
class EnhancedAnalysisEngine:
    """增强版分析引擎 - 在原有基础上增加牌点分析"""
    
    @staticmethod
    def comprehensive_analysis_with_enhancement(sequence, games_with_cards):
        """增强版综合分析 - 保持原有逻辑，只做增强"""
        
        # 1. 原有分析（完全不变）
        original_analysis = UltimateAnalysisEngine.comprehensive_analysis(sequence)
        
        # 2. 牌点增强分析（新增）
        card_enhancement = CardEnhancementAnalyzer.analyze_card_enhancement(games_with_cards)
        
        # 3. 融合决策（不改变原决策，只做微调）
        enhanced_analysis = EnhancedAnalysisEngine._enhance_decision(
            original_analysis, card_enhancement
        )
        
        return enhanced_analysis
    
    @staticmethod
    def _enhance_decision(original_analysis, card_enhancement):
        """增强决策 - 不改变方向，只调整置信度"""
        
        # 复制原有分析结果
        enhanced = original_analysis.copy()
        
        # 只在有增强信号时调整
        if card_enhancement['enhancement_factor'] != 0:
            original_confidence = enhanced['confidence']
            enhancement = card_enhancement['enhancement_factor']
            
            # 调整置信度（限制范围）
            new_confidence = original_confidence + enhancement
            new_confidence = max(0.1, min(0.95, new_confidence))
            
            enhanced['confidence'] = new_confidence
            
            # 更新理由
            original_reason = enhanced.get('reason', '')
            enhancement_reason = card_enhancement.get('reason', '')
            
            if enhancement_reason and enhancement_reason != "无增强信号":
                enhanced['reason'] = f"{original_reason} | 牌点:{enhancement_reason}"
                enhanced['card_enhancement'] = {
                    'factor': card_enhancement['enhancement_factor'],
                    'reason': enhancement_reason
                }
        
        return enhanced
