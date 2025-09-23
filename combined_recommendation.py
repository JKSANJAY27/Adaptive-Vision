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

# ============================
# Advanced Color Optimization Engine
# ============================

class AdvancedColorOptimizer:
    """Enhanced optimizer with fuzzy logic and context awareness"""
    
    def __init__(self, 
                 cvd_profile: AdvancedCVDProfile,
                 context: RecommendationContext):
        self.cvd_profile = cvd_profile
        self.context = context
        self.fuzzy_evaluator = FuzzyColorHarmonyEvaluator(cvd_profile)
        self.emotion_mapper = EmotionalColorMapper()
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
    
    def optimize_colors(self,
                        primary_colors: List[Tuple[int, int, int]],
                        n_iterations: int = 3,
                        top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find optimal colors using multi-objective optimization
        """
        results = []
        
        # Define optimization bounds in HSV space
        bounds = [(0, 360), (0, 100), (0, 100)]
        
        # Multi-objective function
        def objective(hsv):
            h, s, v = hsv
            candidate = self._hsv_to_rgb(h, s, v)
            
            scores = []
            
            for primary in primary_colors:
                # 1. Fuzzy harmony score
                fuzzy_score = self.fuzzy_evaluator.evaluate(candidate, primary)
                
                # 2. Context appropriateness
                context_score = self.emotion_mapper.score_for_context(candidate, self.context)
                
                # 3. CVD distinguishability
                cvd_primary = self.cvd_profile.get_perceived_rgb(primary)
                cvd_candidate = self.cvd_profile.get_perceived_rgb(candidate)
                cvd_delta_e = deltaE_CIE2000(cvd_primary, cvd_candidate)
                cvd_score = min(1.0, cvd_delta_e / (self.min_distinguishable_delta_e * 2))
                
                # 4. Texture compatibility (for CVD compensation)
                texture_score = self._evaluate_texture_compatibility(candidate, primary)
                
                # 5. Confidence adjustment
                confidence_factor = 0.7 + self.cvd_profile.color_confidence * 0.3
                
                # Weighted combination
                weights = self._get_adaptive_weights()
                combined_score = (
                    weights['harmony'] * fuzzy_score +
                    weights['context'] * context_score +
                    weights['cvd'] * cvd_score +
                    weights['texture'] * texture_score
                ) * confidence_factor
                
                scores.append(combined_score)
            
            return -np.mean(scores)  # Negative for minimization
        
        # Run optimization with different strategies
        for strategy in range(min(n_iterations, 15)):
            # Vary strategy parameters
            if strategy < 5:
                strategy_name = 'best1bin'
                popsize = 10
            elif strategy < 10:
                strategy_name = 'rand1bin'
                popsize = 8
            else:
                strategy_name = 'currenttobest1bin'
                popsize = 12
            
            result = differential_evolution(
                objective,
                bounds,
                strategy=strategy_name,
                seed=strategy * 42,
                maxiter=100,
                popsize=popsize,
                tol=0.0001,
                mutation=(0.5, 1.5),
                recombination=0.7
            )
            
            # Extract and evaluate result
            h_opt, s_opt, v_opt = result.x
            optimal_color = self._hsv_to_rgb(h_opt, s_opt, v_opt)
            
            # Full evaluation
            evaluation = self._comprehensive_evaluation(optimal_color, primary_colors)
            
            # Check uniqueness
            is_unique = True
            for existing in results:
                if deltaE_CIE2000(optimal_color, existing["rgb"]) < 15:
                    is_unique = False
                    break
            
            if is_unique and evaluation['final_score'] > 0.5:
                results.append(evaluation)
        
        # Sort and filter results
        results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Apply context-specific filtering
        filtered_results = self._apply_context_filters(results)
        
        return filtered_results[:top_k]
    
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
    
    def _comprehensive_evaluation(self,
                                 color: Tuple[int, int, int],
                                 primary_colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Complete evaluation with all metrics"""
        cvd_color = self.cvd_profile.get_perceived_rgb(color)
        
        evaluations = []
        for primary in primary_colors:
            cvd_primary = self.cvd_profile.get_perceived_rgb(primary)
            
            # Calculate all metrics
            fuzzy_harmony = self.fuzzy_evaluator.evaluate(color, primary)
            context_score = self.emotion_mapper.score_for_context(color, self.context)
            
            normal_delta_e = deltaE_CIE2000(color, primary)
            cvd_delta_e = deltaE_CIE2000(cvd_color, cvd_primary)
            
            texture_compat = self._evaluate_texture_compatibility(color, primary)
            
            # Harmony type analysis
            harmony_type = self._identify_harmony_type(color, primary)
            cvd_harmony_type = self._identify_harmony_type(cvd_color, cvd_primary)
            harmony_preserved = (harmony_type == cvd_harmony_type or 
                               cvd_delta_e >= self.min_distinguishable_delta_e)
            
            evaluations.append({
                "primary_rgb": primary,
                "fuzzy_harmony_score": fuzzy_harmony,
                "context_appropriateness": context_score,
                "normal_delta_e": normal_delta_e,
                "cvd_delta_e": cvd_delta_e,
                "texture_compatibility": texture_compat,
                "harmony_type": harmony_type,
                "cvd_harmony_type": cvd_harmony_type,
                "harmony_preserved": harmony_preserved,
                "distinguishable": cvd_delta_e >= self.min_distinguishable_delta_e
            })
        
        # Calculate final score
        weights = self._get_adaptive_weights()
        harmony_component = weights['harmony'] * np.mean([e["fuzzy_harmony_score"] for e in evaluations])
        context_component = weights['context'] * np.mean([e["context_appropriateness"] for e in evaluations])
        cvd_component = weights['cvd'] * np.mean([min(1.0, e["cvd_delta_e"] / (self.min_distinguishable_delta_e * 2)) 
                                                 for e in evaluations])
        texture_component = weights['texture'] * np.mean([e["texture_compatibility"] for e in evaluations])
        final_score = harmony_component + context_component + cvd_component + texture_component
        
        # Confidence calculation with fuzzy logic
        confidence = self._calculate_confidence(evaluations, final_score)
        
        # Get emotional profile
        emotion_profile = self.emotion_mapper.get_color_emotion(color)
        
        return {
            "rgb": color,
            "hex": self._rgb_to_hex(color),
            "hsv": self._rgb_to_hsv(color),
            "final_score": final_score,
            "confidence": confidence,
            "emotion_profile": emotion_profile,
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

# ============================
# Helper functions (from original code)
# ============================

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
) -> List[Dict[str, Any]]:
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
    
    return recommendations

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
    """Create advanced visualization with comprehensive analysis"""
    n_recs = len(recommendations)
    fig, axes = plt.subplots(n_recs, 6, figsize=(24, 4*n_recs))
    
    if n_recs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, rec in enumerate(recommendations):
        # Color swatches (normal and CVD perception)
        ax1, ax2, ax3, ax4, ax5, ax6 = axes[idx]
        
        # 1. Primary colors display
        ax1.set_title(f"Primary Colors", fontweight='bold')
        for i, color in enumerate(primary_colors):
            rect = patches.Rectangle((i * 0.8, 0), 0.7, 1, 
                                   facecolor=[c/255.0 for c in color], 
                                   edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            # Add color labels
            ax1.text(i * 0.8 + 0.35, 0.5, f'#{rec["hex"]}' if i == 0 else f'#{primary_colors[i][0]:02X}{primary_colors[i][1]:02X}{primary_colors[i][2]:02X}', 
                    ha='center', va='center', fontsize=8, rotation=90)
        ax1.set_xlim(0, len(primary_colors) * 0.8)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # 2. Recommended color (normal vision)
        ax2.set_title(f"Recommendation #{idx+1}\nNormal Vision", fontweight='bold')
        rect = patches.Rectangle((0, 0), 1, 1, 
                               facecolor=[c/255.0 for c in rec['rgb']], 
                               edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(0.5, 0.5, rec['hex'], ha='center', va='center', 
                fontweight='bold', fontsize=12,
                color='white' if sum(rec['rgb']) < 400 else 'black')
        ax2.text(0.5, 0.2, f"Score: {rec['final_score']:.3f}", 
                ha='center', va='center', fontsize=10,
                color='white' if sum(rec['rgb']) < 400 else 'black')
        ax2.text(0.5, 0.8, f"Conf: {rec['confidence']}", 
                ha='center', va='center', fontsize=10,
                color='white' if sum(rec['rgb']) < 400 else 'black')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_aspect('equal')
        ax2.axis('off')
        
        # 3. CVD perception
        ax3.set_title(f"CVD Perception\n(Simulated)", fontweight='bold')
        cvd_color = rec['cvd_perceived_rgb']
        rect = patches.Rectangle((0, 0), 1, 1, 
                               facecolor=[c/255.0 for c in cvd_color], 
                               edgecolor='black', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(0.5, 0.5, rec['cvd_perceived_hex'], ha='center', va='center', 
                fontweight='bold', fontsize=12,
                color='white' if sum(cvd_color) < 400 else 'black')
        ax3.text(0.5, 0.2, f"Min ΔE: {rec['min_cvd_delta_e']:.1f}", 
                ha='center', va='center', fontsize=10,
                color='white' if sum(cvd_color) < 400 else 'black')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_aspect('equal')
        ax3.axis('off')
        
        # 4. Harmony Analysis
        ax4.set_title("Harmony Analysis", fontweight='bold')
        ax4.axis('off')
        
        # Create harmony breakdown
        harmony_data = []
        labels = []
        for eval_data in rec['evaluations']:
            harmony_type = eval_data['harmony_type']
            harmony_score = eval_data['fuzzy_harmony_score']
            harmony_data.append(harmony_score)
            labels.append(f"{harmony_type}\n{harmony_score:.2f}")
        
        # Text display of harmony info
        y_pos = 0.9
        ax4.text(0.05, y_pos, f"Avg Harmony: {rec['avg_fuzzy_harmony']:.2f}", 
                fontsize=10, fontweight='bold', transform=ax4.transAxes)
        y_pos -= 0.15
        
        for i, eval_data in enumerate(rec['evaluations'][:3]):  # Show top 3
            primary_hex = '#{:02X}{:02X}{:02X}'.format(*eval_data['primary_rgb'])
            ax4.text(0.05, y_pos, f"vs {primary_hex}:", fontsize=9, transform=ax4.transAxes)
            y_pos -= 0.1
            ax4.text(0.1, y_pos, f"  Type: {eval_data['harmony_type']}", fontsize=8, transform=ax4.transAxes)
            y_pos -= 0.08
            ax4.text(0.1, y_pos, f"  Score: {eval_data['fuzzy_harmony_score']:.2f}", fontsize=8, transform=ax4.transAxes)
            y_pos -= 0.08
            ax4.text(0.1, y_pos, f"  CVD ΔE: {eval_data['cvd_delta_e']:.1f}", fontsize=8, transform=ax4.transAxes)
            y_pos -= 0.12
        
        # 5. Emotional Profile Radar Chart
        ax5.set_title("Emotional Profile", fontweight='bold')
        
        # Get top 6 emotions
        emotions = rec['emotion_profile']
        top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:6]
        
        if len(top_emotions) >= 3:
            # Create mini radar chart
            angles = np.linspace(0, 2*np.pi, len(top_emotions), endpoint=False).tolist()
            values = [emotion[1] for emotion in top_emotions]
            emotion_labels = [emotion[0][:6] for emotion in top_emotions]  # Truncate labels
            
            # Close the plot
            angles += angles[:1]
            values += values[:1]
            
            ax5 = plt.subplot(n_recs, 6, idx*6 + 5, projection='polar')
            ax5.plot(angles, values, 'o-', linewidth=2, color=[c/255.0 for c in rec['rgb']])
            ax5.fill(angles, values, alpha=0.25, color=[c/255.0 for c in rec['rgb']])
            ax5.set_xticks(angles[:-1])
            ax5.set_xticklabels(emotion_labels, fontsize=8)
            ax5.set_ylim(0, 1)
            ax5.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax5.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=6)
            ax5.grid(True)
            ax5.set_title("Emotional Profile", fontweight='bold', pad=20)
        else:
            # Fallback to text display if not enough emotions
            ax5.axis('off')
            y_pos = 0.9
            ax5.text(0.5, y_pos, "Top Emotions:", ha='center', fontsize=10, 
                    fontweight='bold', transform=ax5.transAxes)
            y_pos -= 0.15
            for emotion, value in top_emotions:
                ax5.text(0.5, y_pos, f"{emotion}: {value:.2f}", ha='center', 
                        fontsize=9, transform=ax5.transAxes)
                y_pos -= 0.12
        
        # 6. Context & CVD Analysis
        ax6.set_title("Analysis Summary", fontweight='bold')
        ax6.axis('off')
        
        # Summary statistics
        y_pos = 0.95
        
        # Context match
        context_score = rec['context_match']
        ax6.text(0.05, y_pos, f"Context Match: {context_score:.2f}", 
                fontsize=10, fontweight='bold', transform=ax6.transAxes,
                color='green' if context_score > 0.7 else 'orange' if context_score > 0.5 else 'red')
        y_pos -= 0.12
        
        # CVD suitability
        all_distinguishable = all(e["distinguishable"] for e in rec['evaluations'])
        ax6.text(0.05, y_pos, f"CVD Safe: {'✓' if all_distinguishable else '⚠'}", 
                fontsize=10, fontweight='bold', transform=ax6.transAxes,
                color='green' if all_distinguishable else 'red')
        y_pos -= 0.12
        
        # Fashion style match
        style_text = f"Style: {context.fashion_style.value.title()}"
        ax6.text(0.05, y_pos, style_text, fontsize=9, transform=ax6.transAxes)
        y_pos -= 0.1
        
        # Season match
        if context.season:
            season_text = f"Season: {context.season.value.title()}"
            ax6.text(0.05, y_pos, season_text, fontsize=9, transform=ax6.transAxes)
            y_pos -= 0.1
        
        # Occasion
        occasion_text = f"Occasion: {context.occasion.value.title()}"
        ax6.text(0.05, y_pos, occasion_text, fontsize=9, transform=ax6.transAxes)
        y_pos -= 0.1
        
        # CVD Profile Summary
        y_pos -= 0.05
        ax6.text(0.05, y_pos, "CVD Profile:", fontsize=9, fontweight='bold', transform=ax6.transAxes)
        y_pos -= 0.08
        ax6.text(0.1, y_pos, f"R-G Confusion: {cvd_profile.red_green_confusion:.1%}", 
                fontsize=8, transform=ax6.transAxes)
        y_pos -= 0.07
        ax6.text(0.1, y_pos, f"B-Y Confusion: {cvd_profile.blue_yellow_confusion:.1%}", 
                fontsize=8, transform=ax6.transAxes)
        y_pos -= 0.07
        ax6.text(0.1, y_pos, f"Texture Reliance: {cvd_profile.texture_reliance:.1%}", 
                fontsize=8, transform=ax6.transAxes)
        y_pos -= 0.07
        ax6.text(0.1, y_pos, f"Confidence: {cvd_profile.color_confidence:.1%}", 
                fontsize=8, transform=ax6.transAxes)
        
        # Add recommendation ranking
        ax6.text(0.05, 0.05, f"Rank: #{idx+1}", fontsize=12, fontweight='bold', 
                transform=ax6.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    # Overall figure title
    fig.suptitle(f'Advanced CVD-Aware Fashion Color Recommendations\n'
                f'Context: {context.fashion_style.value} | {context.occasion.value} | '
                f'{context.season.value if context.season else "Any Season"}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
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
    image_path = "uploads/shirt.jpg" # 📸 <<< REPLACE WITH YOUR IMAGE PATH
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

    recommendations = recommend_fashion_colors_advanced(
        primary_colors=primary_colors_for_recommender,
        cvd_type="protanopia",
        severity=0.6,
        context=context,
        top_k=5,
        visualize=True
    )

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