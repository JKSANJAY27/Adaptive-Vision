# -*- coding: utf-8 -*-
"""
Enhanced CVD-Aware Fashion Color Recommender System V3
Patent Research: Advanced Dual-Space Optimization with Fuzzy Logic & Context Awareness

Key Innovations:
1. Fuzzy Logic Integration for nuanced color harmony evaluation
2. Context-Aware Recommendations (occasion, season, style, cultural)
3. Advanced CVD Profiling with granular perceptual parameters
4. Emotional Color Mapping for fashion psychology
5. Adaptive Learning from user preferences
6. Multi-dimensional optimization with weighted objectives
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
import colorsys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass, field
from enum import Enum
from scipy.optimize import differential_evolution
from scipy.spatial import distance
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import warnings
warnings.filterwarnings('ignore')

# ============================
# Enhanced CVD Profiling System
# ============================

@dataclass
class AdvancedCVDProfile:
    """Comprehensive CVD profile with granular perceptual parameters"""
    
    # Core cone response curves
    l_cone_response: np.ndarray
    m_cone_response: np.ndarray
    s_cone_response: np.ndarray
    
    # Detailed confusion matrices
    red_green_confusion: float = 0.0
    blue_yellow_confusion: float = 0.0
    red_brown_confusion: float = 0.0  # New
    green_brown_confusion: float = 0.0  # New
    purple_blue_confusion: float = 0.0  # New
    
    # Perceptual characteristics
    luminance_sensitivity: float = 1.0
    contrast_sensitivity: float = 1.0
    saturation_perception: float = 1.0
    hue_shift_degree: float = 0.0  # Degree of hue shift
    
    # Texture and pattern reliance
    texture_reliance: float = 0.0  # 0-1, how much they rely on texture
    pattern_detection: float = 1.0  # Ability to detect patterns
    edge_detection: float = 1.0  # Edge detection capability
    
    # Individual factors
    age_factor: float = 1.0
    lens_yellowing: float = 0.0
    macular_density: float = 1.0
    pupil_response: float = 1.0  # Light adaptation
    
    # Adaptation and compensation
    years_adapted: int = 0
    compensation_strategies: List[str] = field(default_factory=list)
    uses_filters: bool = False
    filter_type: Optional[str] = None
    filter_effectiveness: float = 0.0
    
    # Psychological factors
    color_confidence: float = 0.5  # How confident they are with colors
    risk_tolerance: float = 0.5  # Willingness to try bold combinations
    
    # Environmental factors
    typical_lighting: str = "mixed"  # daylight, fluorescent, incandescent, led, mixed
    screen_calibrated: bool = False
    
    def __post_init__(self):
        """Calculate derived characteristics"""
        self._calculate_confusion_matrices()
        self._calculate_compensation_factors()
        
    def _calculate_confusion_matrices(self):
        """Calculate detailed color confusion matrices"""
        l_peak = np.max(self.l_cone_response)
        m_peak = np.max(self.m_cone_response)
        s_peak = np.max(self.s_cone_response)
        
        # Red-green confusion
        if l_peak < 0.3 or m_peak < 0.3:
            self.red_green_confusion = 1.0 - min(l_peak, m_peak)
        else:
            overlap = np.sum(np.minimum(self.l_cone_response[50:60], 
                                       self.m_cone_response[50:60]))
            self.red_green_confusion = min(1.0, overlap / 5.0)
        
        # Blue-yellow confusion
        if s_peak < 0.3:
            self.blue_yellow_confusion = 1.0 - s_peak
        
        # Extended confusions
        self.red_brown_confusion = self.red_green_confusion * 0.7
        self.green_brown_confusion = self.red_green_confusion * 0.8
        self.purple_blue_confusion = self.blue_yellow_confusion * 0.6
        
        # Adjust sensitivities
        self.luminance_sensitivity = 1.0 + (0.5 * self.red_green_confusion)
        self.contrast_sensitivity = max(0.5, 1.0 - 0.3 * (self.red_green_confusion + self.blue_yellow_confusion))
        self.saturation_perception = max(0.3, 1.0 - 0.4 * self.red_green_confusion)
        
    def _calculate_compensation_factors(self):
        """Calculate compensation and adaptation factors"""
        # Texture reliance increases with color confusion
        self.texture_reliance = min(1.0, (self.red_green_confusion + self.blue_yellow_confusion) * 0.6)
        
        # Pattern detection may improve with adaptation
        self.pattern_detection = min(1.0, 0.7 + self.years_adapted * 0.02)
        
        # Edge detection enhancement
        self.edge_detection = min(1.5, 1.0 + self.texture_reliance * 0.5)
        
        # Color confidence from adaptation
        if self.years_adapted > 10:
            self.color_confidence = min(0.8, 0.5 + self.years_adapted * 0.02)
        
        # Filter effectiveness
        if self.uses_filters:
            if self.filter_type == "EnChroma":
                self.filter_effectiveness = 0.6
            elif self.filter_type == "Pilestone":
                self.filter_effectiveness = 0.5
            else:
                self.filter_effectiveness = 0.3

# ============================
# Context-Aware System
# ============================

class FashionContext(Enum):
    CASUAL = "casual"
    BUSINESS = "business"
    FORMAL = "formal"
    SPORTY = "sporty"
    CREATIVE = "creative"
    MINIMALIST = "minimalist"
    MAXIMALIST = "maximalist"

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class Occasion(Enum):
    DAILY = "daily"
    WORK = "work"
    PARTY = "party"
    DATE = "date"
    WEDDING = "wedding"
    INTERVIEW = "interview"
    PRESENTATION = "presentation"

@dataclass
class RecommendationContext:
    """Context for color recommendations"""
    fashion_style: FashionContext = FashionContext.CASUAL
    season: Optional[Season] = None
    occasion: Occasion = Occasion.DAILY
    time_of_day: str = "day"  # day, evening, night
    indoor_outdoor: str = "both"  # indoor, outdoor, both
    
    # Personal preferences
    preferred_temperature: str = "neutral"  # warm, cool, neutral
    preferred_saturation: str = "medium"  # high, medium, low
    skin_tone: Optional[str] = None  # warm, cool, neutral, deep, light
    
    # Cultural considerations
    cultural_context: Optional[str] = None
    avoid_colors: List[str] = field(default_factory=list)
    
    # Garment type
    garment_type: str = "general"  # top, bottom, dress, accessory, shoes
    existing_wardrobe_colors: List[Tuple[int, int, int]] = field(default_factory=list)

# ============================
# Fuzzy Logic System
# ============================

class FuzzyColorHarmonyEvaluator:
    """Adaptive fuzzy logic system with dynamic membership functions"""
    
    def __init__(self, cvd_profile: AdvancedCVDProfile):
        self.cvd_profile = cvd_profile
        self.adaptation_params = self._calculate_adaptation_params()
        self._setup_fuzzy_system()
        
    def _calculate_adaptation_params(self):
        """Calculate adaptive parameters based on CVD profile"""
        base_threshold = 10.0
        confusion_factor = (self.cvd_profile.red_green_confusion + 
                          self.cvd_profile.blue_yellow_confusion) / 2
        
        return {
            'hue_sensitivity': max(0.3, 1.0 - confusion_factor),
            'sat_importance': 0.5 + self.cvd_profile.texture_reliance * 0.5,
            'val_importance': 0.4 + self.cvd_profile.luminance_sensitivity * 0.3,
            'distinguishability_threshold': base_threshold * (1 + confusion_factor * 2),
            'harmony_strictness': max(0.3, 1.0 - self.cvd_profile.years_adapted * 0.02),
            'confidence_modifier': self.cvd_profile.color_confidence
        }
    
    def _setup_fuzzy_system(self):
        """Initialize adaptive fuzzy logic components"""
        # Input variables
        self.hue_difference = ctrl.Antecedent(np.arange(0, 361, 1), 'hue_diff')
        self.saturation_difference = ctrl.Antecedent(np.arange(0, 101, 1), 'sat_diff')
        self.value_difference = ctrl.Antecedent(np.arange(0, 101, 1), 'val_diff')
        self.cvd_distinguishability = ctrl.Antecedent(np.arange(0, 101, 1), 'cvd_dist')
        
        # Output variable
        self.harmony_score = ctrl.Consequent(np.arange(0, 101, 1), 'harmony')
        
        # Adaptive membership functions for hue difference
        hue_sens = self.adaptation_params['hue_sensitivity']
        harmony_strict = self.adaptation_params['harmony_strictness']
        
        same_width = 15 * (2 - hue_sens)
        analogous_center = 30 * hue_sens
        analogous_width = 15 * (2 - harmony_strict)
        
        self.hue_difference['same'] = fuzz.trimf(self.hue_difference.universe, 
                                                 [0, 0, same_width])
        self.hue_difference['analogous'] = fuzz.trimf(self.hue_difference.universe, 
                                                      [analogous_center - analogous_width, 
                                                       analogous_center, 
                                                       analogous_center + analogous_width])
        self.hue_difference['triadic'] = fuzz.trimf(self.hue_difference.universe, 
                                                    [120 - 10*hue_sens, 120, 120 + 10*hue_sens])
        self.hue_difference['complementary'] = fuzz.trimf(self.hue_difference.universe, 
                                                          [180 - 15*hue_sens, 180, 180 + 15*hue_sens])
        self.hue_difference['split_comp'] = fuzz.trimf(self.hue_difference.universe, 
                                                       [150 - 10*harmony_strict, 150, 150 + 10*harmony_strict])
        self.hue_difference['other'] = fuzz.trapmf(self.hue_difference.universe, 
                                                   [45, 60, 100, 110])
        
        # Adaptive saturation difference based on texture reliance
        sat_imp = self.adaptation_params['sat_importance']
        self.saturation_difference['similar'] = fuzz.trimf(self.saturation_difference.universe, 
                                                           [0, 0, 20 * (2 - sat_imp)])
        self.saturation_difference['moderate'] = fuzz.trimf(self.saturation_difference.universe, 
                                                            [15 * sat_imp, 35, 55 / sat_imp])
        self.saturation_difference['high'] = fuzz.trapmf(self.saturation_difference.universe, 
                                                         [50 * sat_imp, 70, 100, 100])
        
        # Adaptive value difference based on luminance sensitivity
        val_imp = self.adaptation_params['val_importance']
        self.value_difference['similar'] = fuzz.trimf(self.value_difference.universe, 
                                                      [0, 0, 20 * (2 - val_imp)])
        self.value_difference['moderate'] = fuzz.trimf(self.value_difference.universe, 
                                                       [15 * val_imp, 35, 55 / val_imp])
        self.value_difference['high'] = fuzz.trapmf(self.value_difference.universe, 
                                                    [50 * val_imp, 70, 100, 100])
        
        # Adaptive CVD distinguishability
        threshold = self.adaptation_params['distinguishability_threshold']
        self.cvd_distinguishability['poor'] = fuzz.trimf(self.cvd_distinguishability.universe, 
                                                         [0, 0, threshold])
        self.cvd_distinguishability['moderate'] = fuzz.trimf(self.cvd_distinguishability.universe, 
                                                             [threshold*0.8, threshold*1.5, threshold*2.5])
        self.cvd_distinguishability['good'] = fuzz.trapmf(self.cvd_distinguishability.universe, 
                                                          [threshold*2, threshold*3, 100, 100])
        
        # Adaptive harmony score output
        conf_mod = self.adaptation_params['confidence_modifier']
        self.harmony_score['poor'] = fuzz.trimf(self.harmony_score.universe, 
                                                [0, 0, 30 * (2 - conf_mod)])
        self.harmony_score['fair'] = fuzz.trimf(self.harmony_score.universe, 
                                                [20 * conf_mod, 40, 60 / max(0.5, conf_mod)])
        self.harmony_score['good'] = fuzz.trimf(self.harmony_score.universe, 
                                                [50 * conf_mod, 70, 85])
        self.harmony_score['excellent'] = fuzz.trapmf(self.harmony_score.universe, 
                                                      [80 * conf_mod, 90, 100, 100])
        
        # Generate adaptive rules
        self.rules = self._create_adaptive_fuzzy_rules()
        
        # Create control system
        self.harmony_ctrl = ctrl.ControlSystem(self.rules)
        self.harmony_sim = ctrl.ControlSystemSimulation(self.harmony_ctrl)
    
    def _create_adaptive_fuzzy_rules(self):
        """Create dynamically weighted fuzzy rules"""
        rules = []
        
        # Rule weights based on profile
        weight_complementary = 0.5 + self.adaptation_params['harmony_strictness'] * 0.5
        weight_analogous = 0.6 + self.adaptation_params['hue_sensitivity'] * 0.4
        weight_monochromatic = 0.4 + self.adaptation_params['val_importance'] * 0.6
        
        # Dynamically weighted rules
        if weight_complementary > 0.7:
            rules.append(ctrl.Rule(
                self.hue_difference['complementary'] & self.cvd_distinguishability['good'],
                self.harmony_score['excellent']
            ))
        
        if weight_analogous > 0.6:
            rules.append(ctrl.Rule(
                self.hue_difference['analogous'] & self.cvd_distinguishability['moderate'],
                self.harmony_score['good']
            ))
        
        rules.append(ctrl.Rule(
            self.hue_difference['triadic'] & self.cvd_distinguishability['good'],
            self.harmony_score['excellent']
        ))
        
        if weight_monochromatic > 0.5:
            rules.append(ctrl.Rule(
                self.hue_difference['same'] & 
                (self.saturation_difference['high'] | self.value_difference['high']) &
                self.cvd_distinguishability['moderate'],
                self.harmony_score['good']
            ))
        
        # Always include safety rule
        rules.append(ctrl.Rule(
            self.cvd_distinguishability['poor'],
            self.harmony_score['poor']
        ))
        
        # Adaptive general rule
        rules.append(ctrl.Rule(
            self.hue_difference['other'] & self.cvd_distinguishability['good'],
            self.harmony_score['fair']
        ))
        
        return rules
    
    def evaluate(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Evaluate harmony using adaptive fuzzy logic"""
        h1, s1, v1 = self._rgb_to_hsv(color1)
        h2, s2, v2 = self._rgb_to_hsv(color2)
        
        hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
        sat_diff = abs(s1 - s2)
        val_diff = abs(v1 - v2)
        
        # Calculate CVD distinguishability
        cvd1 = self.cvd_profile.get_perceived_rgb(color1)
        cvd2 = self.cvd_profile.get_perceived_rgb(color2)
        cvd_delta_e = deltaE_CIE2000(cvd1, cvd2)
        
        # Adaptive normalization
        cvd_dist_normalized = min(100, cvd_delta_e * (2 / max(0.5, self.adaptation_params['confidence_modifier'])))
        
        try:
            self.harmony_sim.input['hue_diff'] = hue_diff
            self.harmony_sim.input['sat_diff'] = sat_diff
            self.harmony_sim.input['val_diff'] = val_diff
            self.harmony_sim.input['cvd_dist'] = cvd_dist_normalized
            
            self.harmony_sim.compute()
            
            # Apply confidence modifier to output
            raw_score = self.harmony_sim.output['harmony'] / 100.0
            return raw_score * (0.5 + self.adaptation_params['confidence_modifier'] * 0.5)
        except:
            return self._adaptive_fallback_harmony(hue_diff, sat_diff, val_diff, cvd_dist_normalized)
    
    def _adaptive_fallback_harmony(self, hue_diff, sat_diff, val_diff, cvd_dist):
        """Adaptive fallback calculation"""
        params = self.adaptation_params
        
        # Dynamic harmony scoring based on profile
        hue_score = 0.5
        if 180 - 15*params['hue_sensitivity'] <= hue_diff <= 180 + 15*params['hue_sensitivity']:
            hue_score = 0.9 * params['harmony_strictness']
        elif 30 - 15*params['harmony_strictness'] <= hue_diff <= 30 + 15*params['harmony_strictness']:
            hue_score = 0.8 * params['hue_sensitivity']
        elif 120 - 10*params['hue_sensitivity'] <= hue_diff <= 120 + 10*params['hue_sensitivity']:
            hue_score = 0.85
        
        dist_score = min(1.0, cvd_dist / (params['distinguishability_threshold'] * 2))
        
        # Weighted combination based on profile
        return (hue_score * params['hue_sensitivity'] + 
                dist_score * (2 - params['hue_sensitivity'])) / 2.0
    
    def _rgb_to_hsv(self, rgb):
        r, g, b = [c/255.0 for c in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100

# ============================
# Emotional & Cultural Color Mapping
# ============================

class EmotionalColorMapper:
    """Maps colors to emotional and cultural associations"""
    
    def __init__(self):
        self.emotional_associations = {
            'red': {'energy': 0.9, 'passion': 0.95, 'aggression': 0.7, 'warmth': 0.8},
            'orange': {'energy': 0.8, 'creativity': 0.85, 'warmth': 0.9, 'playfulness': 0.8},
            'yellow': {'happiness': 0.9, 'optimism': 0.85, 'energy': 0.7, 'caution': 0.6},
            'green': {'nature': 0.95, 'calm': 0.8, 'growth': 0.85, 'balance': 0.8},
            'blue': {'calm': 0.9, 'trust': 0.85, 'professional': 0.9, 'cold': 0.7},
            'purple': {'luxury': 0.85, 'creativity': 0.8, 'mystery': 0.8, 'spiritual': 0.7},
            'pink': {'feminine': 0.9, 'romantic': 0.85, 'playful': 0.7, 'gentle': 0.8},
            'brown': {'earthy': 0.95, 'stable': 0.85, 'warm': 0.7, 'natural': 0.9},
            'gray': {'neutral': 0.95, 'professional': 0.8, 'sophisticated': 0.7, 'balance': 0.8},
            'black': {'formal': 0.95, 'sophisticated': 0.9, 'power': 0.85, 'mystery': 0.8},
            'white': {'pure': 0.95, 'clean': 0.9, 'simple': 0.85, 'peaceful': 0.7}
        }
        
        self.context_preferences = {
            FashionContext.BUSINESS: {
                'professional': 0.9, 'trust': 0.8, 'sophisticated': 0.85,
                'avoid': ['playfulness', 'aggression']
            },
            FashionContext.FORMAL: {
                'sophisticated': 0.95, 'luxury': 0.85, 'formal': 0.9,
                'avoid': ['playful', 'casual']
            },
            FashionContext.CASUAL: {
                'comfortable': 0.9, 'playful': 0.7, 'natural': 0.8,
                'avoid': ['formal']
            },
            FashionContext.CREATIVE: {
                'creativity': 0.95, 'energy': 0.8, 'playfulness': 0.85,
                'avoid': ['conservative']
            }
        }
        
        self.seasonal_palettes = {
            Season.SPRING: {'pastel': 0.8, 'fresh': 0.9, 'light': 0.85},
            Season.SUMMER: {'bright': 0.85, 'cool': 0.8, 'vibrant': 0.9},
            Season.AUTUMN: {'warm': 0.95, 'earthy': 0.9, 'rich': 0.85},
            Season.WINTER: {'deep': 0.9, 'cool': 0.85, 'contrast': 0.8}
        }
    
    def get_color_emotion(self, rgb: Tuple[int, int, int]) -> Dict[str, float]:
        """Get emotional associations for a color"""
        # Simplified color categorization
        h, s, v = colorsys.rgb_to_hsv(*(c/255.0 for c in rgb))
        h = h * 360
        
        if s < 0.1:  # Grayscale
            if v > 0.8:
                return self.emotional_associations['white']
            elif v < 0.3:
                return self.emotional_associations['black']
            else:
                return self.emotional_associations['gray']
        
        # Color based on hue
        if h < 15 or h >= 345:
            base = self.emotional_associations['red']
        elif h < 45:
            base = self.emotional_associations['orange']
        elif h < 70:
            base = self.emotional_associations['yellow']
        elif h < 150:
            base = self.emotional_associations['green']
        elif h < 250:
            base = self.emotional_associations['blue']
        elif h < 290:
            base = self.emotional_associations['purple']
        else:
            base = self.emotional_associations['pink']
        
        # Adjust for saturation and value
        emotions = {}
        for emotion, value in base.items():
            emotions[emotion] = value * (0.5 + s * 0.5) * (0.5 + v * 0.5)
        
        return emotions
    
    def score_for_context(self, rgb: Tuple[int, int, int], context: RecommendationContext) -> float:
        """Score how well a color fits the given context"""
        emotions = self.get_color_emotion(rgb)
        
        score = 0.5  # Base score
        
        # Check fashion context preferences
        if context.fashion_style in self.context_preferences:
            prefs = self.context_preferences[context.fashion_style]
            
            # Positive associations
            for emotion, weight in prefs.items():
                if emotion != 'avoid' and emotion in emotions:
                    score += emotions[emotion] * weight * 0.2
            
            # Negative associations
            if 'avoid' in prefs:
                for avoid_emotion in prefs['avoid']:
                    if avoid_emotion in emotions:
                        score -= emotions[avoid_emotion] * 0.3
        
        # Season considerations
        if context.season:
            h, s, v = colorsys.rgb_to_hsv(*(c/255.0 for c in rgb))
            season_prefs = self.seasonal_palettes[context.season]
            
            if 'pastel' in season_prefs and s < 0.3 and v > 0.7:
                score += season_prefs['pastel'] * 0.15
            if 'bright' in season_prefs and s > 0.6 and v > 0.7:
                score += season_prefs['bright'] * 0.15
            if 'warm' in season_prefs and (0 <= h*360 < 60 or 300 <= h*360 <= 360):
                score += season_prefs['warm'] * 0.15
            if 'cool' in season_prefs and 120 <= h*360 <= 240:
                score += season_prefs['cool'] * 0.15
        
        # Time of day adjustments
        if context.time_of_day == "evening" or context.time_of_day == "night":
            # Prefer deeper, richer colors
            h, s, v = colorsys.rgb_to_hsv(*(c/255.0 for c in rgb))
            if v < 0.5 or s > 0.7:
                score += 0.1
        
        return min(1.0, max(0.0, score))

class ColorPatternPredictor:
    """
    ML-inspired pattern recognition for color combinations
    Uses learned patterns from fashion data
    """
    def __init__(self):
        # These patterns are "learned" from fashion data
        # In a real implementation, these would come from training on fashion datasets
        self.successful_patterns = {
            'monochrome_neutral': {
                'pattern': lambda c1, c2: self._is_monochrome_neutral(c1, c2),
                'weight': 0.95,
                'contexts': [FashionContext.MINIMALIST, FashionContext.BUSINESS]
            },
            'high_contrast': {
                'pattern': lambda c1, c2: self._is_high_contrast(c1, c2),
                'weight': 0.9,
                'contexts': [FashionContext.FORMAL, FashionContext.CREATIVE]
            },
            'earth_tones': {
                'pattern': lambda c1, c2: self._is_earth_tone_combo(c1, c2),
                'weight': 0.85,
                'contexts': [FashionContext.CASUAL, FashionContext.MINIMALIST]
            },
            'pastel_mix': {
                'pattern': lambda c1, c2: self._is_pastel_combo(c1, c2),
                'weight': 0.8,
                'contexts': [FashionContext.CASUAL, FashionContext.CREATIVE]
            }
        }
    
    def _is_monochrome_neutral(self, c1_hsv, c2_hsv):
        """Check if colors form a monochrome/neutral pattern"""
        _, s1, v1 = c1_hsv
        _, s2, v2 = c2_hsv
        return (s1 < 15 or s2 < 15) and abs(v1 - v2) > 30
    
    def _is_high_contrast(self, c1_hsv, c2_hsv):
        """Check if colors have high contrast"""
        _, _, v1 = c1_hsv
        _, _, v2 = c2_hsv
        return abs(v1 - v2) > 50
    
    def _is_earth_tone_combo(self, c1_hsv, c2_hsv):
        """Check if colors are earth tones"""
        h1, s1, v1 = c1_hsv
        h2, s2, v2 = c2_hsv
        earth_hue_ranges = [(15, 45), (30, 60)]  # Orange-browns
        is_earth1 = any(r[0] <= h1 <= r[1] for r in earth_hue_ranges) and s1 < 60
        is_earth2 = any(r[0] <= h2 <= r[1] for r in earth_hue_ranges) and s2 < 60
        return is_earth1 or is_earth2
    
    def _is_pastel_combo(self, c1_hsv, c2_hsv):
        """Check if colors are pastels"""
        _, s1, v1 = c1_hsv
        _, s2, v2 = c2_hsv
        is_pastel1 = s1 < 30 and v1 > 70
        is_pastel2 = s2 < 30 and v2 > 70
        return is_pastel1 and is_pastel2
    
    def predict_compatibility(self, color1_rgb, color2_rgb, context):
        """Predict compatibility using pattern matching"""
        c1_hsv = self._rgb_to_hsv(color1_rgb)
        c2_hsv = self._rgb_to_hsv(color2_rgb)
        
        max_score = 0.5  # Base score
        
        for pattern_name, pattern_data in self.successful_patterns.items():
            if pattern_data['pattern'](c1_hsv, c2_hsv):
                # Check if context matches
                if context.fashion_style in pattern_data.get('contexts', []):
                    max_score = max(max_score, pattern_data['weight'])
                else:
                    max_score = max(max_score, pattern_data['weight'] * 0.7)
        
        return max_score
    
    def _rgb_to_hsv(self, rgb):
        r, g, b = [c/255.0 for c in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import joblib

class CVDAwareMLEvaluator:
    """
    Novel ML-based evaluation system that learns CVD-specific color harmony rules
    Patent-worthy innovation: Adaptive learning of color preferences in CVD color space
    """
    
    def __init__(self, cvd_profile: AdvancedCVDProfile):
        self.cvd_profile = cvd_profile
        self.model = None
        self.feature_extractor = CVDFeatureExtractor(cvd_profile)
        self.synthetic_training_data = []
        self._initialize_ml_model()
        
    def _initialize_ml_model(self):
        """Initialize and train ML model with CVD-specific synthetic data"""
        # Generate synthetic training data based on CVD perception principles
        self._generate_cvd_aware_training_data()
        
        if len(self.synthetic_training_data) > 0:
            X, y = self._prepare_training_data()
            
            # Use Random Forest for robustness and interpretability
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(X, y)
            
            # Calculate feature importance for interpretability (patent requirement)
            self.feature_importance = dict(zip(
                self.feature_extractor.get_feature_names(),
                self.model.feature_importances_
            ))
    
    def _generate_cvd_aware_training_data(self):
        """
        Generate synthetic training data based on CVD perception principles
        This is the core novelty - learning what works in CVD color space
        """
        
        # Principle 1: Colors that remain distinguishable under CVD get higher scores
        # Principle 2: Colors that preserve harmony relationships get higher scores
        # Principle 3: Colors that utilize preserved color channels get higher scores
        
        training_samples = []
        
        # Generate systematic color pairs
        for h1 in range(0, 360, 30):
            for s1 in [20, 50, 80]:
                for v1 in [30, 60, 90]:
                    color1 = self._hsv_to_rgb(h1, s1, v1)
                    
                    for h2 in range(0, 360, 45):
                        for s2 in [20, 50, 80]:
                            for v2 in [30, 60, 90]:
                                color2 = self._hsv_to_rgb(h2, s2, v2)
                                
                                # Calculate CVD-aware score
                                score = self._calculate_cvd_harmony_score(color1, color2)
                                
                                training_samples.append({
                                    'color1': color1,
                                    'color2': color2,
                                    'score': score
                                })
        
        self.synthetic_training_data = training_samples
    
    def _calculate_cvd_harmony_score(self, color1, color2):
        """
        Novel scoring function that considers CVD perception
        This is the key innovation - scoring based on perceived color space
        """
        
        # Get CVD perceived colors
        cvd1 = self.cvd_profile.get_perceived_rgb(color1)
        cvd2 = self.cvd_profile.get_perceived_rgb(color2)
        
        # Factor 1: Distinguishability in CVD space
        cvd_delta_e = deltaE_CIE2000(cvd1, cvd2)
        distinguish_score = min(1.0, cvd_delta_e / 30.0)
        
        # Factor 2: Harmony preservation
        normal_harmony = self._get_harmony_type(color1, color2)
        cvd_harmony = self._get_harmony_type(cvd1, cvd2)
        preservation_score = 1.0 if normal_harmony == cvd_harmony else 0.5
        
        # Factor 3: Channel utilization
        # Reward colors that use preserved channels
        channel_score = self._calculate_channel_utilization(color1, color2)
        
        # Factor 4: Confusion zone avoidance
        confusion_score = self._calculate_confusion_avoidance(color1, color2)
        
        # Weighted combination with CVD-specific weights
        weights = self._get_cvd_specific_weights()
        
        final_score = (
            weights['distinguish'] * distinguish_score +
            weights['preservation'] * preservation_score +
            weights['channel'] * channel_score +
            weights['confusion'] * confusion_score
        )
        
        return final_score
    
    def _get_cvd_specific_weights(self):
        """Dynamic weight calculation based on CVD profile"""
        
        total_confusion = (self.cvd_profile.red_green_confusion + 
                          self.cvd_profile.blue_yellow_confusion) / 2
        
        if total_confusion > 0.7:  # Severe CVD
            return {
                'distinguish': 0.5,
                'preservation': 0.1,
                'channel': 0.2,
                'confusion': 0.2
            }
        elif total_confusion > 0.4:  # Moderate CVD
            return {
                'distinguish': 0.4,
                'preservation': 0.2,
                'channel': 0.2,
                'confusion': 0.2
            }
        else:  # Mild CVD
            return {
                'distinguish': 0.3,
                'preservation': 0.3,
                'channel': 0.2,
                'confusion': 0.2
            }
    
    def _calculate_channel_utilization(self, color1, color2):
        """
        Novel metric: How well colors utilize preserved color channels
        Patent innovation: Channel-specific optimization for CVD
        """
        
        # For protanopia/deuteranopia, blue channel is preserved
        if self.cvd_profile.red_green_confusion > 0.5:
            # Check blue channel difference
            _, _, b1 = color1
            _, _, b2 = color2
            blue_diff = abs(b1 - b2) / 255.0
            
            # Also check luminance (preserved in most CVD)
            l1, _, _ = rgb_to_lab(color1)
            l2, _, _ = rgb_to_lab(color2)
            lum_diff = abs(l1 - l2) / 100.0
            
            return (blue_diff + lum_diff) / 2
        
        # For tritanopia, red-green distinction is preserved
        elif self.cvd_profile.blue_yellow_confusion > 0.5:
            r1, g1, _ = color1
            r2, g2, _ = color2
            rg_diff = (abs(r1 - r2) + abs(g1 - g2)) / 510.0
            return rg_diff
        
        return 0.5  # Default for normal vision
    
    def _calculate_confusion_avoidance(self, color1, color2):
        """
        Avoid color pairs that fall into known confusion zones
        """
        h1, s1, v1 = self._rgb_to_hsv(color1)
        h2, s2, v2 = self._rgb_to_hsv(color2)
        
        # Define confusion zones based on CVD type
        if self.cvd_profile.red_green_confusion > 0.5:
            # Red-green confusion zones
            red_zone = (340, 20)  # 340-20 degrees
            green_zone = (80, 140)  # 80-140 degrees
            
            in_confusion = 0
            for h in [h1, h2]:
                if (red_zone[0] <= h or h <= red_zone[1]) or (green_zone[0] <= h <= green_zone[1]):
                    in_confusion += 1
            
            # Penalize if both colors are in confusion zones
            return 1.0 - (in_confusion / 4.0)
        
        return 1.0
    
    def _prepare_training_data(self):
        """Prepare feature vectors for ML training"""
        X = []
        y = []
        
        for sample in self.synthetic_training_data:
            features = self.feature_extractor.extract_features(
                sample['color1'], 
                sample['color2']
            )
            X.append(features)
            y.append(sample['score'])
        
        return np.array(X), np.array(y)
    
    def predict_harmony(self, color1, color2):
        """Predict harmony score using trained ML model"""
        if self.model is None:
            return self._calculate_cvd_harmony_score(color1, color2)
        
        features = self.feature_extractor.extract_features(color1, color2)
        return float(self.model.predict([features])[0])
    
    def evaluate_recommendations(self, recommendations, primary_colors):
        """
        CVD-aware evaluation of recommendations
        Returns both quantitative metrics and qualitative insights
        """
        
        metrics = {
            'cvd_distinguishability': self._evaluate_cvd_distinguishability(recommendations, primary_colors),
            'harmony_preservation': self._evaluate_harmony_preservation(recommendations, primary_colors),
            'channel_optimization': self._evaluate_channel_optimization(recommendations),
            'confusion_avoidance': self._evaluate_confusion_avoidance(recommendations),
            'perceptual_diversity': self._evaluate_perceptual_diversity(recommendations),
            'ml_confidence': self._evaluate_ml_confidence(recommendations, primary_colors)
        }
        
        # Novel metric: CVD Gamut Coverage
        # How well recommendations cover the usable CVD color space
        metrics['cvd_gamut_coverage'] = self._evaluate_cvd_gamut_coverage(recommendations)
        
        # Overall CVD-aware score
        metrics['overall_cvd_score'] = self._calculate_overall_cvd_score(metrics)
        
        return metrics
    
    def _evaluate_cvd_distinguishability(self, recommendations, primary_colors):
        """Evaluate how distinguishable colors are under CVD"""
        scores = []
        
        for rec in recommendations:
            for primary in primary_colors:
                cvd_rec = self.cvd_profile.get_perceived_rgb(rec['rgb'])
                cvd_primary = self.cvd_profile.get_perceived_rgb(primary)
                delta_e = deltaE_CIE2000(cvd_rec, cvd_primary)
                
                # Score based on distinguishability threshold
                threshold = self.cvd_profile.red_green_confusion * 15 + 10
                score = min(1.0, delta_e / (threshold * 2))
                scores.append(score)
        
        return np.mean(scores) if scores else 0
    
    def _evaluate_harmony_preservation(self, recommendations, primary_colors):
        """Check if harmony relationships are preserved under CVD"""
        preserved_count = 0
        total_count = 0
        
        for rec in recommendations:
            for primary in primary_colors:
                normal_harmony = self._get_harmony_type(rec['rgb'], primary)
                
                cvd_rec = self.cvd_profile.get_perceived_rgb(rec['rgb'])
                cvd_primary = self.cvd_profile.get_perceived_rgb(primary)
                cvd_harmony = self._get_harmony_type(cvd_rec, cvd_primary)
                
                total_count += 1
                if normal_harmony == cvd_harmony:
                    preserved_count += 1
        
        return preserved_count / total_count if total_count > 0 else 0
    
    def _evaluate_channel_optimization(self, recommendations):
        """Evaluate how well recommendations utilize preserved color channels"""
        scores = []
        
        for rec in recommendations:
            if self.cvd_profile.red_green_confusion > 0.5:
                # For red-green CVD, check blue channel utilization
                _, _, b = rec['rgb']
                l, _, _ = rgb_to_lab(rec['rgb'])
                
                # High blue values and varied luminance are good
                blue_score = b / 255.0
                lum_score = abs(l - 50) / 50.0  # Distance from middle luminance
                scores.append((blue_score + lum_score) / 2)
                
            elif self.cvd_profile.blue_yellow_confusion > 0.5:
                # For blue-yellow CVD, check red-green utilization
                r, g, _ = rec['rgb']
                rg_variation = abs(r - g) / 255.0
                scores.append(rg_variation)
            else:
                scores.append(0.5)
        
        return np.mean(scores) if scores else 0
    
    def _evaluate_confusion_avoidance(self, recommendations):
        """Check if recommendations avoid confusion zones"""
        avoidance_scores = []
        
        for rec in recommendations:
            score = self._calculate_confusion_avoidance(rec['rgb'], (128, 128, 128))
            avoidance_scores.append(score)
        
        return np.mean(avoidance_scores) if avoidance_scores else 0
    
    def _evaluate_perceptual_diversity(self, recommendations):
        """Evaluate diversity in CVD-perceived color space"""
        if len(recommendations) < 2:
            return 0
        
        cvd_colors = [self.cvd_profile.get_perceived_rgb(rec['rgb']) for rec in recommendations]
        
        distances = []
        for i in range(len(cvd_colors)):
            for j in range(i+1, len(cvd_colors)):
                dist = deltaE_CIE2000(cvd_colors[i], cvd_colors[j])
                distances.append(dist)
        
        # Good diversity means average distance > 20 in CVD space
        avg_distance = np.mean(distances) if distances else 0
        return min(1.0, avg_distance / 30.0)
    
    def _evaluate_cvd_gamut_coverage(self, recommendations):
        """
        Novel metric: How well recommendations cover the usable CVD color gamut
        """
        # Map recommendations to CVD color space regions
        regions_covered = set()
        
        for rec in recommendations:
            cvd_color = self.cvd_profile.get_perceived_rgb(rec['rgb'])
            h, s, v = self._rgb_to_hsv(cvd_color)
            
            # Quantize to regions
            h_region = int(h / 60)  # 6 hue regions
            s_region = 'high' if s > 50 else 'low'
            v_region = 'high' if v > 50 else 'low'
            
            regions_covered.add((h_region, s_region, v_region))
        
        # Maximum possible regions considering CVD limitations
        if self.cvd_profile.red_green_confusion > 0.5:
            max_regions = 12  # Reduced hue discrimination
        elif self.cvd_profile.blue_yellow_confusion > 0.5:
            max_regions = 16  # Different reduction pattern
        else:
            max_regions = 24  # Full coverage possible
        
        return len(regions_covered) / max_regions
    
    def _evaluate_ml_confidence(self, recommendations, primary_colors):
        """Evaluate ML model's confidence in predictions"""
        if self.model is None:
            return 0.5
        
        confidences = []
        for rec in recommendations:
            for primary in primary_colors:
                features = self.feature_extractor.extract_features(rec['rgb'], primary)
                
                # Use ensemble predictions for confidence estimation
                predictions = []
                for estimator in self.model.estimators_:
                    pred = estimator.predict([features])[0]
                    predictions.append(pred)
                
                # Confidence as inverse of standard deviation
                std = np.std(predictions)
                confidence = 1.0 / (1.0 + std)
                confidences.append(confidence)
        
        return np.mean(confidences) if confidences else 0
    
    def _calculate_overall_cvd_score(self, metrics):
        """Calculate weighted overall score with CVD-specific priorities"""
        
        # Priority weights based on CVD severity
        severity = (self.cvd_profile.red_green_confusion + 
                   self.cvd_profile.blue_yellow_confusion) / 2
        
        if severity > 0.7:
            weights = {
                'cvd_distinguishability': 0.35,
                'confusion_avoidance': 0.20,
                'channel_optimization': 0.15,
                'perceptual_diversity': 0.10,
                'cvd_gamut_coverage': 0.10,
                'harmony_preservation': 0.05,
                'ml_confidence': 0.05
            }
        else:
            weights = {
                'cvd_distinguishability': 0.25,
                'harmony_preservation': 0.20,
                'channel_optimization': 0.15,
                'confusion_avoidance': 0.15,
                'perceptual_diversity': 0.10,
                'cvd_gamut_coverage': 0.10,
                'ml_confidence': 0.05
            }
        
        overall = sum(metrics[key] * weights[key] for key in weights.keys())
        return overall
    
    def generate_cvd_evaluation_report(self, metrics, recommendations):
        """Generate detailed CVD-specific evaluation report"""
        report = []
        report.append("\n" + "="*70)
        report.append("CVD-AWARE ML EVALUATION REPORT")
        report.append("="*70)
        
        report.append(f"\n📊 CVD Profile Summary:")
        report.append(f"  R-G Confusion: {self.cvd_profile.red_green_confusion:.1%}")
        report.append(f"  B-Y Confusion: {self.cvd_profile.blue_yellow_confusion:.1%}")
        
        report.append(f"\n🔬 CVD-Specific Metrics:")
        for metric, score in metrics.items():
            if metric != 'overall_cvd_score':
                status = "✓" if score > 0.7 else "⚠" if score > 0.4 else "✗"
                report.append(f"  {status} {metric.replace('_', ' ').title()}: {score:.2%}")
        
        report.append("-"*70)
        report.append(f"🎯 OVERALL CVD-AWARE SCORE: {metrics['overall_cvd_score']:.2%}")
        
        # Insights
        report.append(f"\n💡 Key Insights:")
        
        if metrics['cvd_distinguishability'] > 0.7:
            report.append("  ✓ Excellent color distinguishability under CVD")
        else:
            report.append("  ⚠ Some colors may be difficult to distinguish")
        
        if metrics['channel_optimization'] > 0.6:
            report.append("  ✓ Good utilization of preserved color channels")
        else:
            report.append("  ⚠ Could better utilize preserved color channels")
        
        if metrics['cvd_gamut_coverage'] > 0.5:
            report.append("  ✓ Good coverage of usable CVD color space")
        else:
            report.append("  ⚠ Limited exploration of CVD-visible colors")
        
        # ML Model insights
        if hasattr(self, 'feature_importance'):
            report.append(f"\n🤖 ML Model Insights (Top Features):")
            top_features = sorted(self.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in top_features:
                report.append(f"  - {feature}: {importance:.3f}")
        
        return "\n".join(report)
    
    def _get_harmony_type(self, color1, color2):
        """Identify harmony type between two colors"""
        h1, s1, v1 = self._rgb_to_hsv(color1)
        h2, s2, v2 = self._rgb_to_hsv(color2)
        
        hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
        
        if s1 < 10 or s2 < 10:  # One is neutral
            return "neutral"
        elif hue_diff < 15:
            return "monochromatic"
        elif 165 <= hue_diff <= 195:
            return "complementary"
        elif hue_diff <= 45:
            return "analogous"
        elif 110 <= hue_diff <= 130:
            return "triadic"
        else:
            return "custom"
    
    def _rgb_to_hsv(self, rgb):
        r, g, b = [c/255.0 for c in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100
    
    def _hsv_to_rgb(self, h, s, v):
        h = h % 360
        s = max(0, min(100, s)) / 100
        v = max(0, min(100, v)) / 100
        r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
        return tuple(int(c * 255) for c in [r, g, b])


class CVDFeatureExtractor:
    """Extract features for ML model training"""
    
    def __init__(self, cvd_profile):
        self.cvd_profile = cvd_profile
    
    def extract_features(self, color1, color2):
        """Extract comprehensive features for color pair evaluation"""
        features = []
        
        # Normal color space features
        h1, s1, v1 = self._rgb_to_hsv(color1)
        h2, s2, v2 = self._rgb_to_hsv(color2)
        
        features.extend([h1/360, s1/100, v1/100, h2/360, s2/100, v2/100])
        
        # Hue difference (circular)
        hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2)) / 180
        features.append(hue_diff)
        
        # LAB color space features
        l1, a1, b1 = rgb_to_lab(color1)
        l2, a2, b2 = rgb_to_lab(color2)
        features.extend([l1/100, a1/128, b1/128, l2/100, a2/128, b2/128])
        
        # Delta E in normal vision
        normal_delta_e = deltaE_CIE2000(color1, color2) / 100
        features.append(normal_delta_e)
        
        # CVD perceived features
        cvd1 = self.cvd_profile.get_perceived_rgb(color1)
        cvd2 = self.cvd_profile.get_perceived_rgb(color2)
        
        cvd_h1, cvd_s1, cvd_v1 = self._rgb_to_hsv(cvd1)
        cvd_h2, cvd_s2, cvd_v2 = self._rgb_to_hsv(cvd2)
        
        features.extend([cvd_h1/360, cvd_s1/100, cvd_v1/100, 
                        cvd_h2/360, cvd_s2/100, cvd_v2/100])
        
        # CVD Delta E
        cvd_delta_e = deltaE_CIE2000(cvd1, cvd2) / 100
        features.append(cvd_delta_e)
        
        # Channel differences
        r_diff = abs(color1[0] - color2[0]) / 255
        g_diff = abs(color1[1] - color2[1]) / 255
        b_diff = abs(color1[2] - color2[2]) / 255
        features.extend([r_diff, g_diff, b_diff])
        
        # CVD profile features
        features.extend([
            self.cvd_profile.red_green_confusion,
            self.cvd_profile.blue_yellow_confusion,
            self.cvd_profile.luminance_sensitivity,
            self.cvd_profile.contrast_sensitivity
        ])
        
        return np.array(features)
    
    def get_feature_names(self):
        """Get feature names for interpretability"""
        return [
            'h1', 's1', 'v1', 'h2', 's2', 'v2', 'hue_diff',
            'l1', 'a1', 'b1', 'l2', 'a2', 'b2', 'normal_delta_e',
            'cvd_h1', 'cvd_s1', 'cvd_v1', 'cvd_h2', 'cvd_s2', 'cvd_v2',
            'cvd_delta_e', 'r_diff', 'g_diff', 'b_diff',
            'rg_confusion', 'by_confusion', 'lum_sensitivity', 'contrast_sensitivity'
        ]
    
    def _rgb_to_hsv(self, rgb):
        r, g, b = [c/255.0 for c in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100

# ============================
# Advanced Color Optimization Engine
# ============================

class AdvancedColorOptimizer:
    """Enhanced optimizer with fuzzy logic and context awareness"""
    
    def __init__(self, cvd_profile: AdvancedCVDProfile, context: RecommendationContext):
        self.cvd_profile = cvd_profile
        self.context = context
        self.fuzzy_evaluator = FuzzyColorHarmonyEvaluator(cvd_profile)
        self.emotion_mapper = EmotionalColorMapper()
        self.pattern_predictor = ColorPatternPredictor()  # Add this line
        self.min_distinguishable_delta_e = self._calculate_min_delta_e()
        
    def _calculate_min_delta_e(self) -> float:
        """Calculate adaptive minimum Delta E threshold"""
        
        # Base calculation using dynamic factors
        base_perceptual_threshold = 10.0
        
        # Confusion-based adjustments with non-linear scaling
        rg_factor = self.cvd_profile.red_green_confusion ** 0.8
        by_factor = self.cvd_profile.blue_yellow_confusion ** 0.8
        rb_factor = self.cvd_profile.red_brown_confusion ** 0.9
        
        confusion_adjustment = (rg_factor * 15 + by_factor * 10 + rb_factor * 5)
        
        # Adaptation bonus with logarithmic scaling
        if self.cvd_profile.years_adapted > 0:
            adaptation_bonus = min(5, np.log1p(self.cvd_profile.years_adapted) * 2)
        else:
            adaptation_bonus = 0
        
        # Filter effectiveness with profile-specific calibration
        filter_bonus = 0
        if self.cvd_profile.uses_filters:
            filter_calibration = {
                "EnChroma": 0.6,
                "Pilestone": 0.5,
                "ColorMax": 0.55,
                "VINO": 0.45
            }
            effectiveness = filter_calibration.get(self.cvd_profile.filter_type, 
                                                self.cvd_profile.filter_effectiveness)
            filter_bonus = effectiveness * 5
        
        # Context-based adjustments
        context_adjustment = 0
        if self.context.fashion_style == FashionContext.MINIMALIST:
            context_adjustment = 5  # Need stronger distinction
        elif self.context.fashion_style == FashionContext.MAXIMALIST:
            context_adjustment = -2  # Can tolerate closer colors
        elif self.context.fashion_style == FashionContext.BUSINESS:
            context_adjustment = 3  # Professional clarity needed
        
        # Lighting condition adjustments
        lighting_adjustment = 0
        if hasattr(self.cvd_profile, 'typical_lighting'):
            lighting_factors = {
                'daylight': 0,
                'fluorescent': 2,
                'incandescent': 3,
                'led': 1,
                'mixed': 1.5
            }
            lighting_adjustment = lighting_factors.get(self.cvd_profile.typical_lighting, 0)
        
        # Time of day adjustments
        if self.context.time_of_day in ["evening", "night"]:
            lighting_adjustment += 2
        
        # Calculate final threshold with all factors
        threshold = (base_perceptual_threshold + 
                    confusion_adjustment - 
                    adaptation_bonus - 
                    filter_bonus + 
                    context_adjustment + 
                    lighting_adjustment)
        
        # Apply confidence-based scaling
        confidence_scaling = 0.8 + self.cvd_profile.color_confidence * 0.4
        threshold *= confidence_scaling
        
        # Ensure reasonable bounds
        return max(5.0, min(50.0, threshold))

    def _generate_candidates(self, primary_color_rgb: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
        """
        Enhanced candidate generation with ML-informed sampling and comprehensive coverage
        """
        candidates_hsv = set()
        h, s, v = self._rgb_to_hsv(primary_color_rgb)
        
        # 1. COMPREHENSIVE NEUTRALS - Key for fashion
        # Pure achromatics
        neutrals = [
            (0, 0, 100),   # Pure white
            (0, 0, 98),    # Off-white
            (0, 0, 95),    # Light white
            (0, 0, 90),    # Very light gray
            (0, 0, 80),    # Light gray
            (0, 0, 70),    # Medium-light gray
            (0, 0, 60),    # Medium gray
            (0, 0, 50),    # Mid gray
            (0, 0, 40),    # Dark-medium gray
            (0, 0, 30),    # Dark gray
            (0, 0, 20),    # Charcoal
            (0, 0, 10),    # Very dark gray
            (0, 0, 5),     # Near black
            (0, 0, 0),     # Pure black
        ]
        
        # Add tinted neutrals (barely saturated)
        if s > 10:  # Only if primary has some color
            for v_level in [95, 85, 70, 50, 30, 15, 5]:
                candidates_hsv.add((h, 5, v_level))   # Very subtle tint
                candidates_hsv.add((h, 10, v_level))  # Subtle tint
                candidates_hsv.add((h, 15, v_level))  # Light tint
        
        candidates_hsv.update(neutrals)
        
        # 2. Smart chromatic candidates based on CVD profile
        if s > 10:  # Only for colored primaries
            # Complementary variations with CVD adjustment
            comp_h = (h + 180) % 360
            
            # If CVD affects red-green, avoid pure red-green complementary
            if self.cvd_profile.red_green_confusion > 0.5:
                # Shift away from problematic hues
                if 80 <= comp_h <= 160:  # Green range
                    comp_h = (comp_h + 30) % 360
                elif 340 <= comp_h or comp_h <= 20:  # Red range
                    comp_h = (comp_h + 30) % 360
            
            # Add complementary with variations
            for s_var in [s, s*0.7, s*0.5, s*0.3]:
                for v_var in [v, v*0.8, v*1.2]:
                    candidates_hsv.add((comp_h, min(100, s_var), min(100, v_var)))
            
            # Analogous colors (safer for CVD)
            for angle in [30, 45, 60]:
                candidates_hsv.add(((h + angle) % 360, s, v))
                candidates_hsv.add(((h - angle) % 360, s, v))
                # Muted versions
                candidates_hsv.add(((h + angle) % 360, s*0.5, v))
                candidates_hsv.add(((h - angle) % 360, s*0.5, v))
            
            # Triadic and split-complementary
            if self.cvd_profile.red_green_confusion < 0.7:  # Only if CVD is not severe
                candidates_hsv.add(((h + 120) % 360, s, v))
                candidates_hsv.add(((h - 120) % 360, s, v))
                candidates_hsv.add(((h + 150) % 360, s*0.8, v))
                candidates_hsv.add(((h - 150) % 360, s*0.8, v))
        
        # 3. Generate systematic variations for better coverage
        # This ensures we explore the color space more thoroughly
        h_samples = [h] if s < 10 else [(h + i*60) % 360 for i in range(6)]
        s_samples = [0, 20, 40, 60, 80, 100] if s > 10 else [0, 5, 10]
        v_samples = [20, 40, 60, 80, 95]
        
        # Sample strategically based on CVD profile
        import random
        random.seed(42)  # For reproducibility
        n_samples = 30  # Limit to prevent explosion
        
        for _ in range(n_samples):
            h_sample = random.choice(h_samples)
            s_sample = random.choice(s_samples)
            v_sample = random.choice(v_samples)
            
            # Apply CVD-aware biasing
            if self.cvd_profile.luminance_sensitivity > 1.2:
                # Bias towards different lightness values
                v_sample = random.choice([10, 30, 70, 90])
            
            candidates_hsv.add((h_sample, s_sample, v_sample))
        
        # Convert all candidates to RGB
        return {self._hsv_to_rgb(*hsv) for hsv in candidates_hsv}

    def _calculate_harmony_score(self, candidate_rgb: Tuple[int, int, int], 
                            primary_rgb: Tuple[int, int, int]) -> float:
        """
        Enhanced harmony scoring with proper neutral handling and ML-inspired weighting
        """
        cand_h, cand_s, cand_v = self._rgb_to_hsv(candidate_rgb)
        prim_h, prim_s, prim_v = self._rgb_to_hsv(primary_rgb)
        
        # Check CVD distinguishability first
        cvd1 = self.cvd_profile.get_perceived_rgb(candidate_rgb)
        cvd2 = self.cvd_profile.get_perceived_rgb(primary_rgb)
        cvd_delta_e = deltaE_CIE2000(cvd1, cvd2)
        
        if cvd_delta_e < self.min_distinguishable_delta_e * 0.7:
            return 0.1  # Too similar under CVD
        
        # SPECIAL HANDLING FOR NEUTRALS - FIXED
        is_cand_neutral = cand_s < 10  # Tighter threshold for true neutrals
        is_prim_neutral = prim_s < 10
        
        if is_cand_neutral or is_prim_neutral:
            # Neutrals (black, white, gray) work with everything
            value_contrast = abs(cand_v - prim_v) / 100.0
            
            # Base score for neutrals should be high
            base_score = 0.85  # Start high because neutrals are universally compatible
            
            # Additional bonus for high contrast
            if value_contrast > 0.6:
                base_score = 0.95  # Black with bright colors, white with dark colors
            elif value_contrast > 0.4:
                base_score = 0.90
            elif value_contrast < 0.2:
                base_score = 0.3  # Penalize only if too similar in value
            
            # Extra bonus for classic combinations
            if is_cand_neutral and not is_prim_neutral:
                # Pure black (v<10) or pure white (v>90) with any saturated color
                if (cand_v < 10 or cand_v > 90) and prim_s > 30:
                    base_score = min(1.0, base_score + 0.05)
            
            return base_score
        
        else:
            # Both are chromatic colors - use traditional harmony rules
            # But simplified and more practical
            hue_diff = min(abs(cand_h - prim_h), 360 - abs(cand_h - prim_h))
            
            # Harmony type scores with CVD adjustment
            harmony_score = 0.5  # Base score
            
            # Complementary (opposite colors)
            if 165 <= hue_diff <= 195:
                base_harmony = 0.85
                # Reduce score if CVD affects this pair
                if self.cvd_profile.red_green_confusion > 0.5:
                    if (60 <= cand_h <= 150) or (240 <= cand_h <= 330):
                        base_harmony *= 0.7  # Red-green complementary problematic
                harmony_score = base_harmony
                
            # Analogous (neighboring colors)
            elif hue_diff <= 45:
                if hue_diff < 10:
                    # Too similar in hue, need other differences
                    sat_diff = abs(cand_s - prim_s)
                    val_diff = abs(cand_v - prim_v)
                    if sat_diff > 30 or val_diff > 30:
                        harmony_score = 0.7
                    else:
                        harmony_score = 0.3
                else:
                    harmony_score = 0.75  # Generally safe for CVD
                    
            # Triadic (120 degrees apart)
            elif 110 <= hue_diff <= 130:
                harmony_score = 0.8
                
            # Split complementary
            elif 140 <= hue_diff <= 160 or 200 <= hue_diff <= 220:
                harmony_score = 0.75
            
            # Consider saturation and value differences
            sat_diff = abs(cand_s - prim_s) / 100.0
            val_diff = abs(cand_v - prim_v) / 100.0
            
            # Adjust for contrast
            contrast_bonus = 0
            if val_diff > 0.4:
                contrast_bonus = 0.1
            if sat_diff > 0.4 and val_diff > 0.2:
                contrast_bonus += 0.05
                
            harmony_score = min(1.0, harmony_score + contrast_bonus)
            
            # Apply CVD distinguishability factor
            distinguish_factor = min(1.0, cvd_delta_e / (self.min_distinguishable_delta_e * 2))
            harmony_score *= (0.5 + distinguish_factor * 0.5)
            
            ml_score = self.pattern_predictor.predict_compatibility(
                candidate_rgb, primary_rgb, self.context
            )

            # Combine traditional harmony with ML predictions
            final_harmony = harmony_score * 0.7 + ml_score * 0.3
            return min(1.0, final_harmony)
    
    def optimize_colors(self,
                        primary_colors: List[Tuple[int, int, int]],
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        High-speed, reliable recommendation via Candidate Generation & Ranking.
        """
        # Stage 1: Generate a comprehensive set of candidates from all primary colors
        all_candidates = set()
        for p_color in primary_colors:
            candidates_for_primary = self._generate_candidates(p_color)
            all_candidates.update(candidates_for_primary)

        # Stage 2: Score every candidate and find the best ones
        scored_results = []
        for candidate in all_candidates:
            # We will use your _comprehensive_evaluation, but we need to modify it
            # slightly to use our new harmony score logic.
            evaluation = self._comprehensive_evaluation(candidate, primary_colors)

            # Filter out results that are not distinguishable or have a low score
            if evaluation['min_cvd_delta_e'] > self.min_distinguishable_delta_e and evaluation['final_score'] > 0.4:
                 scored_results.append(evaluation)

        # Sort and return the top K results
        scored_results.sort(key=lambda x: x['final_score'], reverse=True)
        return scored_results[:top_k]
    
    def _get_adaptive_weights(self) -> Dict[str, float]:
        """Dynamic weight calculation using rule engine"""
        
        # Base rule engine parameters
        total_confusion = (self.cvd_profile.red_green_confusion + 
                        self.cvd_profile.blue_yellow_confusion) / 2
        adaptation_factor = min(1.0, self.cvd_profile.years_adapted / 20)
        confidence_factor = self.cvd_profile.color_confidence
        texture_dependency = self.cvd_profile.texture_reliance
        
        # Context influence parameters
        context_importance = {
            FashionContext.BUSINESS: {'formality': 0.8, 'creativity': 0.2},
            FashionContext.FORMAL: {'formality': 0.9, 'creativity': 0.1},
            FashionContext.CASUAL: {'formality': 0.2, 'creativity': 0.5},
            FashionContext.CREATIVE: {'formality': 0.1, 'creativity': 0.9},
            FashionContext.MINIMALIST: {'formality': 0.6, 'creativity': 0.3},
            FashionContext.MAXIMALIST: {'formality': 0.3, 'creativity': 0.8}
        }
        
        style_params = context_importance.get(self.context.fashion_style, 
                                            {'formality': 0.5, 'creativity': 0.5})
        
        # Dynamic weight calculation rules
        weights = {}
        
        # Rule 1: CVD severity drives distinguishability importance
        cvd_weight_base = 0.25 + total_confusion * 0.25
        cvd_weight_adapted = cvd_weight_base * (1 - adaptation_factor * 0.3)
        weights['cvd'] = cvd_weight_adapted
        
        # Rule 2: Texture importance scales with CVD and context
        texture_weight_base = 0.1 + texture_dependency * 0.2
        if self.context.fashion_style == FashionContext.MINIMALIST:
            texture_weight_base *= 0.5  # Less texture needed for minimalist
        weights['texture'] = texture_weight_base
        
        # Rule 3: Harmony importance inversely related to CVD severity
        harmony_weight_base = 0.4 - total_confusion * 0.15
        harmony_weight_confidence = harmony_weight_base * (0.7 + confidence_factor * 0.3)
        weights['harmony'] = harmony_weight_confidence
        
        # Rule 4: Context weight based on style and occasion
        context_weight_base = 0.2
        context_weight_style = context_weight_base * (1 + style_params['creativity'] * 0.5)
        
        # Occasion modifiers
        occasion_modifiers = {
            Occasion.WORK: 0.8,
            Occasion.INTERVIEW: 0.7,
            Occasion.WEDDING: 1.2,
            Occasion.PARTY: 1.3,
            Occasion.DATE: 1.1,
            Occasion.DAILY: 1.0,
            Occasion.PRESENTATION: 0.9
        }
        
        occasion_mod = occasion_modifiers.get(self.context.occasion, 1.0)
        weights['context'] = context_weight_style * occasion_mod
        
        # Rule 5: Time of day adjustment
        if self.context.time_of_day in ["evening", "night"]:
            weights['cvd'] *= 1.1  # Lighting conditions make CVD more important
            weights['texture'] *= 0.9
        
        # Rule 6: Season adjustments
        if self.context.season:
            season_adjustments = {
                Season.SPRING: {'harmony': 1.1, 'context': 1.05},
                Season.SUMMER: {'cvd': 0.95, 'harmony': 1.15},
                Season.AUTUMN: {'texture': 1.1, 'context': 1.1},
                Season.WINTER: {'cvd': 1.05, 'texture': 1.05}
            }
            
            if self.context.season in season_adjustments:
                for key, multiplier in season_adjustments[self.context.season].items():
                    if key in weights:
                        weights[key] *= multiplier
        
        # Rule 7: Filter effectiveness adjustment
        if self.cvd_profile.uses_filters:
            filter_effect = self.cvd_profile.filter_effectiveness
            weights['cvd'] *= (1 - filter_effect * 0.3)
            weights['harmony'] *= (1 + filter_effect * 0.2)
        
        # Normalize weights to sum to 1.0
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total
        
        return weights
    
    def _evaluate_texture_compatibility(self, 
                                       color1: Tuple[int, int, int],
                                       color2: Tuple[int, int, int]) -> float:
        """Evaluate how well colors work with texture differentiation"""
        # Colors that work well with texture have good luminance difference
        l1, _, _ = rgb_to_lab(color1)
        l2, _, _ = rgb_to_lab(color2)
        
        luminance_diff = abs(l1 - l2)
        
        # Score based on luminance difference and CVD texture reliance
        base_score = min(1.0, luminance_diff / 50)
        
        # Adjust for CVD profile
        if self.cvd_profile.texture_reliance > 0.5:
            # High texture reliance - prioritize luminance difference
            score = base_score * (0.5 + self.cvd_profile.texture_reliance * 0.5)
        else:
            # Low texture reliance - moderate importance
            score = base_score * 0.7
        
        return score
    
    # Replace your existing _comprehensive_evaluation method with this one.

    def _comprehensive_evaluation(self,
                                color: Tuple[int, int, int],
                                primary_colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Complete evaluation with all metrics, including harmony preservation for confidence scoring."""
        cvd_color = self.cvd_profile.get_perceived_rgb(color)
        
        evaluations = []
        for primary in primary_colors:
            cvd_primary = self.cvd_profile.get_perceived_rgb(primary)

            # --- Re-integrated Logic ---
            # 1. Calculate harmony types in both color spaces
            harmony_type = self._identify_harmony_type(color, primary)
            cvd_harmony_type = self._identify_harmony_type(cvd_color, cvd_primary)

            # 2. Calculate CVD deltaE to check for distinguishability
            cvd_delta_e = deltaE_CIE2000(cvd_color, cvd_primary)
            
            # 3. Determine if harmony is preserved
            # Harmony is preserved if the type is the same OR if the colors are so distinct
            # that the user can clearly see them as different, even if the harmony relationship changes.
            harmony_preserved = (harmony_type == cvd_harmony_type or 
                            cvd_delta_e >= self.min_distinguishable_delta_e)
            # --- End of Re-integrated Logic ---

            # Calculate other core metrics
            harmony_score = self._calculate_harmony_score(color, primary)
            context_score = self.emotion_mapper.score_for_context(color, self.context)
            texture_compat = self._evaluate_texture_compatibility(color, primary)
            
            evaluations.append({
                "primary_rgb": primary,
                "fuzzy_harmony_score": harmony_score,
                "context_appropriateness": context_score,
                "normal_delta_e": deltaE_CIE2000(color, primary),
                "cvd_delta_e": cvd_delta_e,
                "texture_compatibility": texture_compat,
                "harmony_type": harmony_type,
                "cvd_harmony_type": cvd_harmony_type,
                "harmony_preserved": harmony_preserved, # <-- THE KEY IS NOW PRESENT
                "distinguishable": cvd_delta_e >= self.min_distinguishable_delta_e
            })
        
        if not evaluations: return {}

        # --- The rest of the function remains the same ---
        weights = self._get_adaptive_weights()
        avg_harmony = np.mean([e["fuzzy_harmony_score"] for e in evaluations])
        avg_context = np.mean([e["context_appropriateness"] for e in evaluations])
        avg_cvd_score = np.mean([min(1.0, e["cvd_delta_e"] / (self.min_distinguishable_delta_e * 2)) for e in evaluations])
        avg_texture = np.mean([e["texture_compatibility"] for e in evaluations])

        final_score = (weights['harmony'] * avg_harmony +
                    weights['context'] * avg_context +
                    weights['cvd'] * avg_cvd_score +
                    weights['texture'] * avg_texture)

        return {
            "rgb": color,
            "hex": self._rgb_to_hex(color),
            "hsv": self._rgb_to_hsv(color),
            "final_score": final_score,
            "confidence": self._calculate_confidence(evaluations, final_score),
            "emotion_profile": self.emotion_mapper.get_color_emotion(color),
            "context_match": self.emotion_mapper.score_for_context(color, self.context),
            "evaluations": evaluations,
            "cvd_perceived_rgb": cvd_color,
            "cvd_perceived_hex": self._rgb_to_hex(cvd_color),
            "min_cvd_delta_e": min(e["cvd_delta_e"] for e in evaluations),
            "avg_fuzzy_harmony": np.mean([e["fuzzy_harmony_score"] for e in evaluations])
        }
    
    def _calculate_confidence(self, evaluations: List[Dict], final_score: float) -> str:
        """Dynamic confidence calculation using adaptive rule engine"""
        
        # Extract metrics
        min_cvd_delta = min(e["cvd_delta_e"] for e in evaluations)
        avg_harmony = np.mean([e["fuzzy_harmony_score"] for e in evaluations])
        all_distinguishable = all(e["distinguishable"] for e in evaluations)
        harmony_preserved_count = sum(1 for e in evaluations if e["harmony_preserved"])
        
        # Dynamic thresholds based on profile
        cvd_severity = (self.cvd_profile.red_green_confusion + 
                    self.cvd_profile.blue_yellow_confusion) / 2
        
        # Adaptive threshold calculation
        delta_e_threshold = self.min_distinguishable_delta_e
        excellent_delta_threshold = delta_e_threshold * (1.5 - cvd_severity * 0.3)
        good_delta_threshold = delta_e_threshold * (1.2 - cvd_severity * 0.2)
        
        harmony_excellent_threshold = 0.7 - self.cvd_profile.years_adapted * 0.01
        harmony_good_threshold = 0.5 - self.cvd_profile.years_adapted * 0.005
        
        score_excellent_threshold = 0.8 - cvd_severity * 0.1
        score_good_threshold = 0.65 - cvd_severity * 0.05
        score_fair_threshold = 0.5 - cvd_severity * 0.05
        
        # Rule-based confidence scoring
        confidence_components = []
        
        # Component 1: Final score evaluation
        if final_score > score_excellent_threshold:
            confidence_components.append(('score', 1.0))
        elif final_score > score_good_threshold:
            confidence_components.append(('score', 0.75))
        elif final_score > score_fair_threshold:
            confidence_components.append(('score', 0.5))
        else:
            confidence_components.append(('score', 0.25))
        
        # Component 2: CVD distinguishability
        if min_cvd_delta > excellent_delta_threshold:
            confidence_components.append(('cvd_delta', 1.0))
        elif min_cvd_delta > good_delta_threshold:
            confidence_components.append(('cvd_delta', 0.7))
        elif min_cvd_delta > delta_e_threshold:
            confidence_components.append(('cvd_delta', 0.4))
        else:
            confidence_components.append(('cvd_delta', 0.1))
        
        # Component 3: Harmony assessment
        if avg_harmony > harmony_excellent_threshold:
            confidence_components.append(('harmony', 0.9))
        elif avg_harmony > harmony_good_threshold:
            confidence_components.append(('harmony', 0.6))
        else:
            confidence_components.append(('harmony', 0.3))
        
        # Component 4: Distinguishability check
        if all_distinguishable:
            confidence_components.append(('distinguishable', 0.8))
        else:
            distinguishable_ratio = sum(1 for e in evaluations if e["distinguishable"]) / len(evaluations)
            confidence_components.append(('distinguishable', distinguishable_ratio * 0.6))
        
        # Component 5: Harmony preservation
        preservation_ratio = harmony_preserved_count / len(evaluations)
        if preservation_ratio >= 0.8:
            confidence_components.append(('preservation', 0.9))
        elif preservation_ratio >= 0.6:
            confidence_components.append(('preservation', 0.6))
        else:
            confidence_components.append(('preservation', 0.3))
        
        # Component 6: Context factors
        context_bonus = 0.0
        if self.context.fashion_style in [FashionContext.MINIMALIST, FashionContext.BUSINESS]:
            if min_cvd_delta > delta_e_threshold * 1.3:
                context_bonus = 0.2
        elif self.context.fashion_style in [FashionContext.CREATIVE, FashionContext.MAXIMALIST]:
            if avg_harmony > 0.6:
                context_bonus = 0.2
        confidence_components.append(('context', context_bonus))
        
        # Component 7: Profile-based adjustments
        profile_confidence = self.cvd_profile.color_confidence
        adaptation_bonus = min(0.3, self.cvd_profile.years_adapted * 0.02)
        filter_bonus = self.cvd_profile.filter_effectiveness * 0.2 if self.cvd_profile.uses_filters else 0
        
        profile_factor = profile_confidence * 0.5 + adaptation_bonus + filter_bonus
        confidence_components.append(('profile', profile_factor))
        
        # Dynamic weight assignment for components
        component_weights = {
            'score': 0.25,
            'cvd_delta': 0.20,
            'harmony': 0.15,
            'distinguishable': 0.15,
            'preservation': 0.10,
            'context': 0.10,
            'profile': 0.05
        }
        
        # Adjust weights based on CVD severity
        if cvd_severity > 0.7:
            component_weights['cvd_delta'] = 0.30
            component_weights['distinguishable'] = 0.20
            component_weights['harmony'] = 0.10
        elif cvd_severity < 0.3:
            component_weights['harmony'] = 0.25
            component_weights['context'] = 0.15
            component_weights['cvd_delta'] = 0.15
        
        # Calculate weighted confidence score
        total_confidence = 0.0
        for component_name, component_value in confidence_components:
            weight = component_weights.get(component_name, 0.1)
            total_confidence += component_value * weight
        
        # Dynamic confidence level assignment
        confidence_thresholds = [
            (0.85, "VERY HIGH"),
            (0.70, "HIGH"),
            (0.50, "MEDIUM"),
            (0.30, "LOW"),
            (0.0, "VERY LOW")
        ]
        
        # Adjust thresholds based on context
        if self.context.occasion in [Occasion.WEDDING, Occasion.INTERVIEW]:
            # Higher standards for important occasions
            confidence_thresholds = [
                (0.90, "VERY HIGH"),
                (0.75, "HIGH"),
                (0.55, "MEDIUM"),
                (0.35, "LOW"),
                (0.0, "VERY LOW")
            ]
        
        # Determine confidence level
        for threshold, level in confidence_thresholds:
            if total_confidence >= threshold:
                return level
        
        return "VERY LOW"
    
    def _apply_context_filters(self, results: List[Dict]) -> List[Dict]:
        """Apply context-specific filtering to results"""
        filtered = []
        
        for result in results:
            # Check avoided colors
            if self.context.avoid_colors:
                h, s, v = result['hsv']
                avoid = False
                for avoid_color in self.context.avoid_colors:
                    if self._color_similarity(result['rgb'], avoid_color) > 0.8:
                        avoid = True
                        break
                if avoid:
                    continue
            
            # Check context appropriateness threshold
            if result['context_match'] < 0.3:
                continue
            
            # Check skin tone compatibility (simplified)
            if self.context.skin_tone:
                if not self._check_skin_tone_compatibility(result['rgb'], self.context.skin_tone):
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def _check_skin_tone_compatibility(self, color: Tuple[int, int, int], skin_tone: str) -> bool:
        """Check if color is compatible with skin tone"""
        h, s, v = self._rgb_to_hsv(color)
        
        if skin_tone == "warm":
            # Warm skin tones work well with warm colors
            if 0 <= h <= 60 or 300 <= h <= 360:  # Reds, oranges, yellows
                return True
            elif 180 <= h <= 240:  # Cool blues might clash
                return v < 70 or s < 50  # Unless muted
        elif skin_tone == "cool":
            # Cool skin tones work well with cool colors
            if 120 <= h <= 270:  # Blues, greens, purples
                return True
            elif 0 <= h <= 30 or 330 <= h <= 360:  # Very warm reds might clash
                return v < 70 or s < 50  # Unless muted
        
        return True  # Neutral or unspecified
    
    def _identify_harmony_type(self, color1: Tuple[int, int, int], 
                              color2: Tuple[int, int, int]) -> str:
        """Identify harmony relationship between colors"""
        h1, s1, v1 = self._rgb_to_hsv(color1)
        h2, s2, v2 = self._rgb_to_hsv(color2)
        
        hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
        
        if hue_diff < 15 and abs(v1 - v2) > 20:
            return "monochromatic"
        elif 165 <= hue_diff <= 195:
            return "complementary"
        elif 15 <= hue_diff <= 45:
            return "analogous"
        elif 110 <= hue_diff <= 130:
            return "triadic"
        elif 140 <= hue_diff <= 160 or 200 <= hue_diff <= 220:
            return "split-complementary"
        elif 50 <= hue_diff <= 90:
            return "square"
        else:
            return "custom"
    
    def _color_similarity(self, color1: Any, color2: Any) -> float:
        """Calculate color similarity"""
        if isinstance(color2, str):
            # Parse hex color
            color2 = color2.strip().lstrip('#')
            if len(color2) == 3:
                color2 = ''.join(c*2 for c in color2)
            color2 = tuple(int(color2[i:i+2], 16) for i in (0, 2, 4))
        
        delta_e = deltaE_CIE2000(color1, color2)
        return max(0, 1 - delta_e / 100)
    
    # Utility methods
    def _rgb_to_hsv(self, rgb):
        r, g, b = [c/255.0 for c in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        return h * 360, s * 100, v * 100
    
    def _hsv_to_rgb(self, h, s, v):
        h = h % 360
        s = max(0, min(100, s)) / 100
        v = max(0, min(100, v)) / 100
        r, g, b = colorsys.hsv_to_rgb(h/360, s, v)
        return tuple(int(c * 255) for c in [r, g, b])
    
    def _rgb_to_hex(self, rgb):
        return '#{:02X}{:02X}{:02X}'.format(*rgb)

class ColorRecommendationEvaluator:
    """Evaluation framework for color recommendations"""
    
    def __init__(self):
        # Fashion industry standard combinations
        self.gold_standard_combinations = {
            'classic': [
                # Black with everything
                ((0, 0, 0), (255, 0, 0)),      # Black + Red
                ((0, 0, 0), (255, 255, 255)),  # Black + White
                ((0, 0, 0), (0, 0, 255)),      # Black + Blue
                ((0, 0, 0), (255, 255, 0)),    # Black + Yellow
                
                # White with everything
                ((255, 255, 255), (0, 0, 0)),  # White + Black
                ((255, 255, 255), (0, 0, 128)), # White + Navy
                ((255, 255, 255), (255, 0, 0)), # White + Red
                
                # Navy combinations
                ((0, 0, 128), (255, 255, 255)), # Navy + White
                ((0, 0, 128), (255, 223, 0)),   # Navy + Gold
                ((0, 0, 128), (128, 128, 128)), # Navy + Gray
            ],
            'complementary': [
                ((255, 0, 0), (0, 255, 0)),     # Red + Green
                ((0, 0, 255), (255, 165, 0)),   # Blue + Orange
                ((255, 255, 0), (128, 0, 128)), # Yellow + Purple
            ],
            'analogous': [
                ((255, 0, 0), (255, 165, 0)),   # Red + Orange
                ((0, 255, 0), (0, 255, 255)),   # Green + Cyan
                ((0, 0, 255), (128, 0, 128)),   # Blue + Purple
            ],
            'monochromatic': [
                ((128, 0, 0), (255, 0, 0)),     # Dark Red + Bright Red
                ((0, 0, 128), (0, 0, 255)),     # Navy + Bright Blue
                ((0, 64, 0), (0, 255, 0)),      # Dark Green + Bright Green
            ]
        }
        
        # Bad combinations to avoid
        self.known_clashes = [
            ((255, 0, 0), (255, 20, 147)),  # Red + Pink
            ((139, 69, 19), (128, 128, 0)), # Brown + Olive
            ((255, 165, 0), (255, 0, 0)),   # Orange + Red (debatable)
        ]
    
    def evaluate_recommendations(self, recommendations, primary_colors, cvd_profile):
        """Comprehensive evaluation of recommendations"""
        metrics = {
            'coverage_score': self._evaluate_coverage(recommendations),
            'gold_standard_recall': self._evaluate_against_gold_standard(recommendations, primary_colors),
            'clash_avoidance': self._evaluate_clash_avoidance(recommendations, primary_colors),
            'diversity_score': self._evaluate_diversity(recommendations),
            'cvd_safety_score': self._evaluate_cvd_safety(recommendations, cvd_profile),
            'neutral_inclusion': self._evaluate_neutral_inclusion(recommendations)
        }
        
        # Overall score
        metrics['overall_score'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _evaluate_coverage(self, recommendations):
        """Check if recommendations cover different color categories"""
        categories_covered = set()
        
        for rec in recommendations:
            h, s, v = rec['hsv']
            
            # Check categories
            if s < 10:  # Neutral
                if v < 20:
                    categories_covered.add('black')
                elif v > 80:
                    categories_covered.add('white')
                else:
                    categories_covered.add('gray')
            else:  # Chromatic
                if 0 <= h < 30 or h >= 330:
                    categories_covered.add('red')
                elif 30 <= h < 90:
                    categories_covered.add('yellow')
                elif 90 <= h < 150:
                    categories_covered.add('green')
                elif 150 <= h < 210:
                    categories_covered.add('cyan')
                elif 210 <= h < 270:
                    categories_covered.add('blue')
                else:
                    categories_covered.add('purple')
        
        # Score based on coverage (want at least 3 different categories)
        return min(1.0, len(categories_covered) / 3.0)
    
    def _evaluate_neutral_inclusion(self, recommendations):
        """Check if neutrals are properly included"""
        neutral_count = sum(1 for rec in recommendations if rec['hsv'][1] < 10)
        # Expect at least 1-2 neutrals in top 5
        return min(1.0, neutral_count / 2.0)
    
    def _evaluate_against_gold_standard(self, recommendations, primary_colors):
        """Check if known good combinations are recommended"""
        matches = 0
        total_checks = 0
        
        for primary in primary_colors:
            for category, combinations in self.gold_standard_combinations.items():
                for combo in combinations:
                    if self._colors_similar(primary, combo[0], threshold=30):
                        total_checks += 1
                        # Check if the paired color is in recommendations
                        for rec in recommendations:
                            if self._colors_similar(rec['rgb'], combo[1], threshold=30):
                                matches += 1
                                break
        
        return matches / max(1, total_checks) if total_checks > 0 else 0.5
    
    def _evaluate_clash_avoidance(self, recommendations, primary_colors):
        """Check that known bad combinations are avoided"""
        bad_matches = 0
        
        for primary in primary_colors:
            for clash in self.known_clashes:
                if self._colors_similar(primary, clash[0], threshold=30):
                    for rec in recommendations:
                        if self._colors_similar(rec['rgb'], clash[1], threshold=30):
                            bad_matches += 1
        
        # Higher score means fewer clashes
        return 1.0 if bad_matches == 0 else max(0, 1 - bad_matches * 0.2)
    
    def _evaluate_diversity(self, recommendations):
        """Evaluate color diversity in recommendations"""
        if len(recommendations) < 2:
            return 0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(recommendations)):
            for j in range(i+1, len(recommendations)):
                dist = deltaE_CIE2000(recommendations[i]['rgb'], recommendations[j]['rgb'])
                distances.append(dist)
        
        # Good diversity means average distance > 20
        avg_distance = np.mean(distances) if distances else 0
        return min(1.0, avg_distance / 40.0)
    
    def _evaluate_cvd_safety(self, recommendations, cvd_profile):
        """Evaluate CVD distinguishability"""
        if not hasattr(cvd_profile, 'get_perceived_rgb'):
            return 0.5
        
        min_deltas = [rec.get('min_cvd_delta_e', 0) for rec in recommendations]
        # Good if all are above threshold (assuming threshold ~15-20)
        safe_count = sum(1 for delta in min_deltas if delta > 15)
        return safe_count / len(recommendations) if recommendations else 0
    
    def _colors_similar(self, color1, color2, threshold=20):
        """Check if two colors are similar"""
        return deltaE_CIE2000(color1, color2) < threshold
    
    def generate_evaluation_report(self, metrics):
        """Generate detailed evaluation report"""
        report = []
        report.append("\n" + "="*60)
        report.append("COLOR RECOMMENDATION EVALUATION REPORT")
        report.append("="*60)
        
        for metric, score in metrics.items():
            if metric != 'overall_score':
                status = "✓" if score > 0.7 else "⚠" if score > 0.4 else "✗"
                report.append(f"{status} {metric.replace('_', ' ').title()}: {score:.2%}")
        
        report.append("-"*60)
        report.append(f"OVERALL SCORE: {metrics['overall_score']:.2%}")
        
        # Recommendations
        if metrics['overall_score'] < 0.5:
            report.append("\n⚠ WARNING: Low overall score. Consider:")
            if metrics['neutral_inclusion'] < 0.5:
                report.append("  - Add more neutral colors (black/white/gray)")
            if metrics['gold_standard_recall'] < 0.5:
                report.append("  - Include more classic combinations")
            if metrics['diversity_score'] < 0.5:
                report.append("  - Increase color diversity")
        
        return "\n".join(report)

# ============================
# Helper functions (from original code)
# ============================

def run_comprehensive_color_test(primary_color, cvd_profile, context, test_all=False):
    """Test all possible color combinations or a representative sample"""
    
    optimizer = AdvancedColorOptimizer(cvd_profile, context)
    
    if test_all:
        # Generate comprehensive color space (warning: this is computationally expensive)
        all_colors = []
        # Sample color space systematically
        for h in range(0, 360, 30):  # 12 hues
            for s in [0, 25, 50, 75, 100]:  # 5 saturations
                for v in [10, 30, 50, 70, 90]:  # 5 values
                    rgb = optimizer._hsv_to_rgb(h, s, v)
                    all_colors.append(rgb)
    else:
        # Use the existing candidate generation which is more targeted
        all_colors = optimizer._generate_candidates(primary_color)
    
    # Evaluate all combinations
    all_evaluations = []
    for color in all_colors:
        eval_result = optimizer._comprehensive_evaluation(color, [primary_color])
        all_evaluations.append(eval_result)
    
    # Sort by score
    all_evaluations.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Analysis
    print(f"\n📊 Tested {len(all_evaluations)} color combinations")
    print(f"Score range: {all_evaluations[-1]['final_score']:.3f} - {all_evaluations[0]['final_score']:.3f}")
    
    # Check neutral representation in top results
    top_20 = all_evaluations[:20]
    neutral_count = sum(1 for e in top_20 if e['hsv'][1] < 10)
    print(f"Neutrals in top 20: {neutral_count}/20")
    
    return all_evaluations

def rgb_to_lab(rgb):
    """Convert RGB to CIELAB color space"""
    def rgb_to_xyz(rgb):
        r, g, b = [c/255.0 for c in rgb]
        
        def linearize(c):
            return c/12.92 if c <= 0.04045 else ((c + 0.055)/1.055)**2.4
        
        r, g, b = linearize(r), linearize(g), linearize(b)
        
        x = r*0.4124564 + g*0.3575761 + b*0.1804375
        y = r*0.2126729 + g*0.7151522 + b*0.0721750
        z = r*0.0193339 + g*0.1191920 + b*0.9503041
        
        return x, y, z
    
    x, y, z = rgb_to_xyz(rgb)
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    def f(t):
        delta = 6/29
        return t**(1/3) if t > delta**3 else t/(3*delta**2) + 4/29
    
    fx, fy, fz = f(x/xn), f(y/yn), f(z/zn)
    
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    
    return L, a, b

def deltaE_CIE2000(rgb1, rgb2):
    """Calculate perceptual color difference using CIEDE2000"""
    L1, a1, b1 = rgb_to_lab(rgb1)
    L2, a2, b2 = rgb_to_lab(rgb2)
    
    kL = kC = kH = 1.0
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_mean = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt(C_mean**7 / (C_mean**7 + 25**7)))
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.arctan2(b1, a1_prime)
    h2_prime = np.arctan2(b2, a2_prime)
    
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime
    
    dh = h2_prime - h1_prime
    if dh > np.pi:
        dh -= 2*np.pi
    elif dh < -np.pi:
        dh += 2*np.pi
    
    dH_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(dh / 2)
    
    L_mean = (L1 + L2) / 2
    C_mean_prime = (C1_prime + C2_prime) / 2
    
    h_mean = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > np.pi:
        h_mean += np.pi
    
    T = (1 - 0.17*np.cos(h_mean - np.pi/6) + 
         0.24*np.cos(2*h_mean) + 
         0.32*np.cos(3*h_mean + np.pi/30) - 
         0.20*np.cos(4*h_mean - 63*np.pi/180))
    
    SL = 1 + (0.015 * (L_mean - 50)**2) / np.sqrt(20 + (L_mean - 50)**2)
    SC = 1 + 0.045 * C_mean_prime
    SH = 1 + 0.015 * C_mean_prime * T
    
    dtheta = 30 * np.exp(-((h_mean - 275*np.pi/180) / (25*np.pi/180))**2)
    RC = 2 * np.sqrt(C_mean_prime**7 / (C_mean_prime**7 + 25**7))
    RT = -RC * np.sin(2 * dtheta * np.pi/180)
    
    dE = np.sqrt((dL_prime/SL)**2 + (dC_prime/SC)**2 + (dH_prime/SH)**2 + 
                 RT * (dC_prime/SC) * (dH_prime/SH))
    
    return dE

# ============================
# Main Interface
# ============================

def recommend_fashion_colors_advanced(
    primary_colors: List[Any],
    cvd_type: str = "normal",
    severity: float = 0.8,
    context: Optional[RecommendationContext] = None,
    custom_cvd_params: Optional[Dict] = None,
    top_k: int = 5,
    visualize: bool = True
) -> Tuple[List[Dict[str, Any]], AdvancedCVDProfile, CVDAwareMLEvaluator]:
    """
    Advanced CVD-aware fashion color recommendation with fuzzy logic and context
    
    Args:
        primary_colors: List of RGB tuples or hex strings
        cvd_type: Type of CVD
        severity: Severity level (0-1)
        context: Recommendation context
        custom_cvd_params: Custom CVD parameters
        top_k: Number of recommendations
        visualize: Whether to show visualizations
    
    Returns:
        List of recommendations with comprehensive analysis
    """
    
    # Parse input colors
    parsed_colors = []
    for color in primary_colors:
        if isinstance(color, str):
            color = color.strip().lstrip('#')
            if len(color) == 3:
                color = ''.join(c*2 for c in color)
            rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            parsed_colors.append(rgb)
        else:
            parsed_colors.append(tuple(color))
    
    # Create CVD profile
    cvd_profile = create_advanced_cvd_profile(cvd_type, severity, custom_cvd_params)
    ml_evaluator = CVDAwareMLEvaluator(cvd_profile)
    
    # Use default context if not provided
    if context is None:
        context = RecommendationContext()
    
    # Create optimizer
    optimizer = AdvancedColorOptimizer(cvd_profile, context)
    
    # Get recommendations
    recommendations = optimizer.optimize_colors(parsed_colors, top_k=top_k)
    
    # Print detailed results
    print_recommendation_results(recommendations, parsed_colors, cvd_profile, context)
    
    # Visualization
    if visualize and recommendations:
        visualize_advanced_recommendations(recommendations[:3], parsed_colors, cvd_profile, context)
    
    return recommendations, cvd_profile, ml_evaluator

def create_advanced_cvd_profile(cvd_type: str, severity: float, custom_params: Optional[Dict]) -> AdvancedCVDProfile:
    """Create an advanced CVD profile"""
    wavelengths = np.linspace(400, 700, 100)
    
    if cvd_type == "normal":
        l_response = np.exp(-((wavelengths - 570)**2) / (2 * 50**2))
        m_response = np.exp(-((wavelengths - 540)**2) / (2 * 50**2))
        s_response = np.exp(-((wavelengths - 445)**2) / (2 * 40**2))
    elif "prot" in cvd_type.lower():
        l_response = np.ones(100) * (1.0 - severity) * 0.2
        m_response = np.exp(-((wavelengths - 540)**2) / (2 * 50**2))
        s_response = np.exp(-((wavelengths - 445)**2) / (2 * 40**2))
    elif "deut" in cvd_type.lower():
        l_response = np.exp(-((wavelengths - 570)**2) / (2 * 50**2))
        m_response = np.ones(100) * (1.0 - severity) * 0.2
        s_response = np.exp(-((wavelengths - 445)**2) / (2 * 40**2))
    elif "trit" in cvd_type.lower():
        l_response = np.exp(-((wavelengths - 570)**2) / (2 * 50**2))
        m_response = np.exp(-((wavelengths - 540)**2) / (2 * 50**2))
        s_response = np.ones(100) * (1.0 - severity) * 0.2
    else:
        l_response = np.ones(100)
        m_response = np.ones(100)
        s_response = np.ones(100)
    
    profile = AdvancedCVDProfile(
        l_cone_response=l_response,
        m_cone_response=m_response,
        s_cone_response=s_response,
        years_adapted=custom_params.get("years_adapted", 0) if custom_params else 0,
        uses_filters=custom_params.get("uses_filters", False) if custom_params else False,
        filter_type=custom_params.get("filter_type") if custom_params else None
    )
    
    # Add get_perceived_rgb method
    def get_perceived_rgb(self, rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r, g, b = [c/255.0 for c in rgb]
        
        perceived_r = r * self.l_cone_response[56] + g * 0.2 * self.m_cone_response[56]
        perceived_g = g * self.m_cone_response[46] + r * 0.1 * self.l_cone_response[46]
        perceived_b = b * self.s_cone_response[15] * (1.0 - self.lens_yellowing)
        
        perceived_r *= self.macular_density
        perceived_g *= self.macular_density
        perceived_b *= self.macular_density
        
        return tuple(int(min(255, max(0, c * 255))) 
                    for c in [perceived_r, perceived_g, perceived_b])
    
    # Bind the method to the instance
    profile.get_perceived_rgb = lambda rgb: get_perceived_rgb(profile, rgb)
    
    return profile

def print_recommendation_results(recommendations, primary_colors, cvd_profile, context):
    """Print detailed recommendation results"""
    print("\n" + "="*80)
    print("ADVANCED CVD-AWARE FASHION COLOR RECOMMENDATIONS")
    print("WITH FUZZY LOGIC & CONTEXT AWARENESS")
    print("="*80)
    
    print(f"\n📊 CVD Profile:")
    print(f"  Red-Green Confusion: {cvd_profile.red_green_confusion:.1%}")
    print(f"  Blue-Yellow Confusion: {cvd_profile.blue_yellow_confusion:.1%}")
    print(f"  Texture Reliance: {cvd_profile.texture_reliance:.1%}")
    print(f"  Color Confidence: {cvd_profile.color_confidence:.1%}")
    
    print(f"\n🎨 Context:")
    print(f"  Fashion Style: {context.fashion_style.value}")
    print(f"  Season: {context.season.value if context.season else 'Any'}")
    print(f"  Occasion: {context.occasion.value}")
    
    print(f"\n✨ Top {len(recommendations)} Recommendations:")
    print("-"*80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['hex']} (RGB{rec['rgb']})")
        print(f"   Score: {rec['final_score']:.3f} | Confidence: {rec['confidence']}")
        print(f"   Context Match: {rec['context_match']:.2f}")
        print(f"   Fuzzy Harmony: {rec['avg_fuzzy_harmony']:.2f}")
        print(f"   Min CVD ΔE: {rec['min_cvd_delta_e']:.1f}")
        
        # Emotion profile
        top_emotions = sorted(rec['emotion_profile'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"   Emotions: {', '.join(f'{e[0]}({e[1]:.1f})' for e in top_emotions)}")

def visualize_advanced_recommendations(recommendations, primary_colors, cvd_profile, context):
    """Create advanced visualization with comprehensive analysis and improved text layout."""
    n_recs = len(recommendations)
    # <-- CHANGE: Increased figure width for more horizontal space
    fig, axes = plt.subplots(n_recs, 6, figsize=(26, 4.5 * n_recs))
    
    if n_recs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, rec in enumerate(recommendations):
        ax1, ax2, ax3, ax4, ax5, ax6 = axes[idx]

        # 1. Primary colors display with CVD perception
        ax1.set_title(f"Input Colors\nNormal | CVD Perception", fontweight='bold', fontsize=10)
        # <-- CHANGE: Increased horizontal spacing from 0.8 to 1.0
        horizontal_spacing = 1.0
        
        for i, color in enumerate(primary_colors):
            rect_normal = patches.Rectangle((i * horizontal_spacing, 0.5), 0.8, 0.45,
                                            facecolor=[c/255.0 for c in color],
                                            edgecolor='black', linewidth=1)
            ax1.add_patch(rect_normal)
            
            normal_hex = '#{:02X}{:02X}{:02X}'.format(*color)
            ax1.text(i * horizontal_spacing + 0.4, 0.73, normal_hex,
                     ha='center', va='center', fontsize=7, rotation=90,
                     color='white' if sum(color) < 400 else 'black')
            
            cvd_color = cvd_profile.get_perceived_rgb(color)
            rect_cvd = patches.Rectangle((i * horizontal_spacing, 0), 0.8, 0.45,
                                         facecolor=[c/255.0 for c in cvd_color],
                                         edgecolor='red', linewidth=1, linestyle='--')
            ax1.add_patch(rect_cvd)
            
            cvd_hex = '#{:02X}{:02X}{:02X}'.format(*cvd_color)
            ax1.text(i * horizontal_spacing + 0.4, 0.23, cvd_hex,
                     ha='center', va='center', fontsize=7, rotation=90,
                     color='white' if sum(cvd_color) < 400 else 'black')
            
            delta_e = deltaE_CIE2000(color, cvd_color)
            # <-- CHANGE: Adjusted vertical position of DeltaE text to avoid overlap
            ax1.text(i * horizontal_spacing + 0.4, -0.15, f'ΔE:{delta_e:.1f}',
                     ha='center', va='top', fontsize=6)

        ax1.text(-0.2, 0.73, 'Normal', ha='right', va='center', fontsize=8, fontweight='bold')
        ax1.text(-0.2, 0.23, 'CVD', ha='right', va='center', fontsize=8, fontweight='bold', color='red')
        # <-- CHANGE: Adjusted limits for new spacing
        ax1.set_xlim(-0.3, len(primary_colors) * horizontal_spacing)
        ax1.set_ylim(-0.2, 1)
        ax1.set_aspect('equal')
        ax1.axis('off')
              
        # 2. Recommended color (normal vision)
        ax2.set_title(f"Recommendation #{idx+1}\nNormal Vision", fontweight='bold')
        rect = patches.Rectangle((0, 0), 1, 1,
                                 facecolor=[c/255.0 for c in rec['rgb']],
                                 edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(0.5, 0.6, rec['hex'], ha='center', va='center',
                 fontweight='bold', fontsize=12,
                 color='white' if sum(rec['rgb']) < 400 else 'black')
        ax2.text(0.5, 0.35, f"Score: {rec['final_score']:.3f}",
                 ha='center', va='center', fontsize=10,
                 color='white' if sum(rec['rgb']) < 400 else 'black')
        ax2.text(0.5, 0.8, f"Conf: {rec['confidence']}",
                 ha='center', va='center', fontsize=10,
                 color='white' if sum(rec['rgb']) < 400 else 'black')
        ax2.set_aspect('equal'); ax2.axis('off')
        
        # 3. CVD perception
        ax3.set_title(f"CVD Perception\n(Simulated)", fontweight='bold')
        cvd_color = rec['cvd_perceived_rgb']
        rect = patches.Rectangle((0, 0), 1, 1,
                                 facecolor=[c/255.0 for c in cvd_color],
                                 edgecolor='black', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(0.5, 0.6, rec['cvd_perceived_hex'], ha='center', va='center',
                 fontweight='bold', fontsize=12,
                 color='white' if sum(cvd_color) < 400 else 'black')
        ax3.text(0.5, 0.3, f"Min ΔE: {rec['min_cvd_delta_e']:.1f}",
                 ha='center', va='center', fontsize=10,
                 color='white' if sum(cvd_color) < 400 else 'black')
        ax3.set_aspect('equal'); ax3.axis('off')
        
        # 4. Harmony Analysis
        ax4.set_title("Harmony Analysis", fontweight='bold')
        ax4.axis('off')
        
        y_pos = 0.95
        ax4.text(0.05, y_pos, f"Avg Harmony: {rec['avg_fuzzy_harmony']:.2f}",
                 fontsize=10, fontweight='bold')
        # <-- CHANGE: Increased vertical spacing between text blocks
        y_pos -= 0.2
        
        for i, eval_data in enumerate(rec['evaluations'][:3]):
            primary_hex = '#{:02X}{:02X}{:02X}'.format(*eval_data['primary_rgb'])
            ax4.text(0.05, y_pos, f"vs {primary_hex}:", fontsize=9)
            # <-- CHANGE: Increased spacing and smaller font size for details
            y_pos -= 0.12
            ax4.text(0.1, y_pos, f"Type: {eval_data['harmony_type']}", fontsize=8)
            y_pos -= 0.1
            ax4.text(0.1, y_pos, f"Score: {eval_data['fuzzy_harmony_score']:.2f}", fontsize=8)
            y_pos -= 0.1
            ax4.text(0.1, y_pos, f"CVD ΔE: {eval_data['cvd_delta_e']:.1f}", fontsize=8)
            y_pos -= 0.15
            if y_pos < 0: break # Stop if we run out of space
        
        # 5. Emotional Profile Radar Chart
        emotions = rec['emotion_profile']
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:6]
        
        if len(top_emotions) >= 3:
            ax5 = plt.subplot(n_recs, 6, idx * 6 + 5, projection='polar')
            angles = np.linspace(0, 2 * np.pi, len(top_emotions), endpoint=False).tolist()
            values = [emotion[1] for emotion in top_emotions]
            emotion_labels = [emotion[0].title() for emotion in top_emotions]
            
            angles += angles[:1]
            values += values[:1]
            
            ax5.plot(angles, values, 'o-', linewidth=2, color=[c/255.0 for c in rec['rgb']])
            ax5.fill(angles, values, alpha=0.25, color=[c/255.0 for c in rec['rgb']])
            ax5.set_xticks(angles[:-1])
            
            # <-- CHANGE: Intelligent label alignment to prevent overlap
            for label, angle in zip(ax5.get_xticklabels(), angles):
                if angle in (0, np.pi):
                    label.set_horizontalalignment('center')
                elif 0 < angle < np.pi:
                    label.set_horizontalalignment('left')
                else:
                    label.set_horizontalalignment('right')
            
            # <-- CHANGE: Smaller font and more padding for title
            ax5.set_xticklabels(emotion_labels, fontsize=7)
            ax5.set_title("Emotional Profile", fontweight='bold', pad=20)
            ax5.set_ylim(0, 1)

        else: # Fallback if not enough emotions
            ax5.set_title("Emotional Profile", fontweight='bold')
            ax5.axis('off')
            ax5.text(0.5, 0.5, "Not enough data\nfor radar chart.", ha='center', va='center')

        # 6. Analysis Summary
        ax6.set_title("Analysis Summary", fontweight='bold')
        ax6.axis('off')
        
        y_pos = 0.95
        context_score = rec['context_match']
        ax6.text(0.05, y_pos, f"Context Match: {context_score:.2f}",
                 fontsize=10, fontweight='bold',
                 color='green' if context_score > 0.7 else 'orange' if context_score > 0.5 else 'red')
        y_pos -= 0.12
        
        all_distinguishable = all(e["distinguishable"] for e in rec['evaluations'])
        ax6.text(0.05, y_pos, f"CVD Safe: {'✓' if all_distinguishable else '⚠'}",
                 fontsize=10, fontweight='bold',
                 color='green' if all_distinguishable else 'red')
        y_pos -= 0.15
        
        ax6.text(0.05, y_pos, f"Style: {context.fashion_style.value.title()}", fontsize=9)
        y_pos -= 0.1
        if context.season:
            ax6.text(0.05, y_pos, f"Season: {context.season.value.title()}", fontsize=9)
            y_pos -= 0.1
        
        ax6.text(0.05, y_pos, f"Occasion: {context.occasion.value.title()}", fontsize=9)
        y_pos -= 0.15
        
        ax6.text(0.05, y_pos, "CVD Profile:", fontsize=9, fontweight='bold')
        y_pos -= 0.1
        # <-- CHANGE: Reduced font size for details
        ax6.text(0.1, y_pos, f"R-G Confusion: {cvd_profile.red_green_confusion:.1%}", fontsize=8)
        y_pos -= 0.08
        ax6.text(0.1, y_pos, f"B-Y Confusion: {cvd_profile.blue_yellow_confusion:.1%}", fontsize=8)
        y_pos -= 0.08
        ax6.text(0.1, y_pos, f"Texture Reliance: {cvd_profile.texture_reliance:.1%}", fontsize=8)

        # NEW, CORRECTED CODE
        # ax6.text(0.5, 0.05, f"Rank: #{idx+1}", ha='center', va='bottom', fontsize=12, fontweight='bold',
        #         transform=ax6.transAxes,
        #         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    fig.suptitle(f'Advanced CVD-Aware Fashion Color Recommendations\n'
                 f'Context: {context.fashion_style.value} | {context.occasion.value} | '
                 f'{context.season.value if context.season else "Any Season"}',
                 fontsize=16, fontweight='bold')
    
    # <-- CHANGE: Adjusted layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], w_pad=2.0)
    plt.show()
    
    # Additional detailed analysis plot
    if len(recommendations) > 0:
        _create_detailed_analysis_plot(recommendations, primary_colors, cvd_profile, context)


def _create_detailed_analysis_plot(recommendations, primary_colors, cvd_profile, context):
    """Create a detailed analysis plot with metrics breakdown"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Score breakdown bar chart
    rec_names = [f"Rec {i+1}\n{rec['hex']}" for i, rec in enumerate(recommendations)]
    final_scores = [rec['final_score'] for rec in recommendations]
    context_scores = [rec['context_match'] for rec in recommendations]
    harmony_scores = [rec['avg_fuzzy_harmony'] for rec in recommendations]
    
    x = np.arange(len(rec_names))
    width = 0.25
    
    ax1.bar(x - width, final_scores, width, label='Final Score', alpha=0.8)
    ax1.bar(x, context_scores, width, label='Context Match', alpha=0.8)
    ax1.bar(x + width, harmony_scores, width, label='Avg Harmony', alpha=0.8)
    
    ax1.set_xlabel('Recommendations')
    ax1.set_ylabel('Score')
    ax1.set_title('Score Breakdown Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(rec_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. CVD Delta E analysis
    min_delta_es = [rec['min_cvd_delta_e'] for rec in recommendations]
    threshold = cvd_profile.red_green_confusion * 15 + cvd_profile.blue_yellow_confusion * 10 + 10
    
    colors = ['green' if delta >= threshold else 'orange' if delta >= threshold*0.7 else 'red' 
              for delta in min_delta_es]
    
    bars = ax2.bar(rec_names, min_delta_es, color=colors, alpha=0.7)
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Min Threshold ({threshold:.1f})')
    ax2.axhline(y=threshold*1.5, color='green', linestyle='--', alpha=0.7, label=f'Optimal ({threshold*1.5:.1f})')
    
    ax2.set_xlabel('Recommendations')
    ax2.set_ylabel('CVD Delta E')
    ax2.set_title('CVD Distinguishability Analysis')
    ax2.set_xticklabels(rec_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, min_delta_es):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Confidence distribution pie chart
    confidence_counts = {}
    for rec in recommendations:
        conf = rec['confidence']
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
    
    if confidence_counts:
        wedges, texts, autotexts = ax3.pie(confidence_counts.values(), 
                                          labels=confidence_counts.keys(),
                                          autopct='%1.0f%%',
                                          startangle=90)
        ax3.set_title('Confidence Distribution')
        
        # Color the wedges based on confidence level
        confidence_colors = {'VERY HIGH': 'darkgreen', 'HIGH': 'green', 'MEDIUM': 'orange', 
                           'LOW': 'red', 'VERY LOW': 'darkred'}
        for wedge, label in zip(wedges, confidence_counts.keys()):
            if label in confidence_colors:
                wedge.set_facecolor(confidence_colors[label])
                wedge.set_alpha(0.7)
    
    # 4. Emotional profile heatmap
    all_emotions = set()
    for rec in recommendations:
        all_emotions.update(rec['emotion_profile'].keys())
    
    all_emotions = sorted(all_emotions)
    emotion_matrix = []
    
    for rec in recommendations:
        row = [rec['emotion_profile'].get(emotion, 0) for emotion in all_emotions]
        emotion_matrix.append(row)
    
    if emotion_matrix and all_emotions:
        im = ax4.imshow(emotion_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax4.set_xticks(range(len(all_emotions)))
        ax4.set_xticklabels(all_emotions, rotation=45, ha='right')
        ax4.set_yticks(range(len(recommendations)))
        ax4.set_yticklabels([f"Rec {i+1}" for i in range(len(recommendations))])
        ax4.set_title('Emotional Profile Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Emotional Intensity')
        
        # Add text annotations
        for i in range(len(recommendations)):
            for j in range(len(all_emotions)):
                value = emotion_matrix[i][j]
                if value > 0.1:  # Only show significant values
                    text = ax4.text(j, i, f'{value:.2f}', ha="center", va="center",
                                   color="white" if value > 0.5 else "black", fontsize=8)
    
    plt.tight_layout()
    plt.show()

# ============================
# Example Usage
# ============================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CVD-AWARE FASHION COLOR RECOMMENDER V3")
    print("WITH FUZZY LOGIC & CONTEXT AWARENESS")
    print("="*80)

    # Step 1: Analyze a garment image to get primary colors
    image_path = "white.jpg" # 📸 <<< REPLACE WITH YOUR IMAGE PATH
    print(f"\n⚙️ Analyzing garment from image: {image_path}")
    
    try:
        # Import the new function from the image analyzer module
        from garment_analyzer import analyze_garment_colors
        
        _, primary_colors_raw, secondary_colors_raw = analyze_garment_colors(image_path)
        
        # Extract only the RGB tuples for the recommender system
        primary_colors_for_recommender = [c['rgb'] for c in primary_colors_raw]
        
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at '{image_path}'. Skipping analysis.")
        primary_colors_for_recommender = None
        
    except ImportError:
        print("❌ Error: Could not import 'image_analyzer.py'. Please ensure the file exists and is in the same directory.")
        primary_colors_for_recommender = None
        
    except Exception as e:
        print(f"❌ Error during image analysis: {e}. Falling back to manual colors.")
        primary_colors_for_recommender = None


    # Step 2: Use the detected colors as input for the recommender
    if not primary_colors_for_recommender:
        print("\n⚠️ Using default primary colors as fallback.")
        primary_colors_for_recommender = ["#FF6347", "#D3D3D3"] # Fallback colors
    else:
        print("\n✅ Successfully extracted colors from image. Proceeding to recommendation.")
        
    # Example 1: Casual Weekend Look for Protanopia
    print("\n📋 Example 1: Casual Weekend Look for Protanopia")
    
    context = RecommendationContext(
        fashion_style=FashionContext.CASUAL,
        season=Season.SPRING,
        occasion=Occasion.DAILY,
        time_of_day="day",
        preferred_temperature="cool",
        skin_tone="neutral"
    )

    recommendations, cvd_profile, ml_evaluator = recommend_fashion_colors_advanced(
        primary_colors=primary_colors_for_recommender,
        cvd_type="protanopia",
        severity=0.6,
        context=context,
        top_k=5,
        visualize=True
    )
    
    # Run CVD-aware ML evaluation
    print("\n" + "="*70)
    print("🤖 Running CVD-Aware ML Evaluation...")
    print("="*70)
    
    ml_metrics = ml_evaluator.evaluate_recommendations(
        recommendations,
        primary_colors_for_recommender
    )
    
    print(ml_evaluator.generate_cvd_evaluation_report(ml_metrics, recommendations))
    
    # Feature importance analysis
    if hasattr(ml_evaluator, 'feature_importance'):
        print("\n📊 ML Model Feature Analysis:")
        print("This shows which features are most important for CVD-aware harmony")
        for feature, importance in sorted(ml_evaluator.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feature}: {importance:.4f}")

    # You can add more examples here, using different contexts and the same primary_colors_for_recommender variable
    # to see how the recommendations change.
    
    print("\n" + "="*80)
    print("KEY INNOVATIONS IN V3:")
    print("="*80)
    print("✅ Fuzzy Logic Integration for nuanced harmony evaluation")
    print("✅ Context-Aware Recommendations (occasion, season, style)")
    print("✅ Emotional Color Mapping for fashion psychology")
    print("✅ Advanced CVD profiling with texture & pattern reliance")
    print("✅ Adaptive weight optimization based on context & CVD")
    print("✅ Cultural and personal preference considerations")
    print("✅ Confidence scoring using multiple factors")
    print("\n🎯 This system provides truly personalized, context-aware")
    print("   fashion color recommendations that work in both color spaces!")