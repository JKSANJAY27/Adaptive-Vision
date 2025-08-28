# -*- coding: utf-8 -*-
"""
Improved Adaptive + Fuzzy Color Recommender
- Enhanced fuzzy system with proper HSV-based fashion rules
- Color tone analysis (NEUTRAL, DARK, BRIGHT)
- Temperature analysis (WARM, COOL)
- Multiple outfit matching schemes (Basic, Analogous, Contrast, etc.)
- CVD-aware recommendations
- Comprehensive visual output and detailed explanations
"""

import math, itertools, json, os, csv
from typing import List, Tuple, Dict, Optional, Any
import colorsys
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Basic color utilities
# -----------------------------
def clamp01(x): return max(0.0, min(1.0, x))
def to01(rgb): return tuple(c/255.0 for c in rgb)
def to255(rgb01): return tuple(int(round(clamp01(c)*255)) for c in rgb01)
def hex_to_rgb(h):
    s = h.strip().lstrip('#')
    if len(s)==3: s = ''.join(ch*2 for ch in s)
    return (int(s[0:2],16), int(s[2:4],16), int(s[4:6],16))
def rgb_to_hex(rgb): return '#{:02X}{:02X}{:02X}'.format(*rgb)

def rgb_to_hsv_deg(rgb):
    r,g,b = to01(rgb)
    h,s,v = colorsys.rgb_to_hsv(r,g,b)
    return (h*360.0, s*100.0, v*100.0)

def hsv_deg_to_rgb(h,s,v):
    h01 = (h%360)/360.0
    s01 = clamp01(s/100.0)
    v01 = clamp01(v/100.0)
    r,g,b = colorsys.hsv_to_rgb(h01,s01,v01)
    return to255((r,g,b))

# XYZ/Lab ΔE76
def srgb_to_linear(c): return c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4
def rgb_to_xyz(rgb):
    r,g,b = to01(rgb)
    rl,gl,bl = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)
    X = rl*0.4124564 + gl*0.3575761 + bl*0.1804375
    Y = rl*0.2126729 + gl*0.7151522 + bl*0.0721750
    Z = rl*0.0193339 + gl*0.1191920 + bl*0.9503041
    return (X,Y,Z)
def xyz_to_lab(xyz):
    Xr, Yr, Zr = 0.95047, 1.00000, 1.08883
    X,Y,Z = xyz
    def f(t): return t**(1/3) if t>0.008856 else (7.787*t + 16/116)
    fx,fy,fz = f(X/Xr), f(Y/Yr), f(Z/Zr)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return (L,a,b)
def rgb_to_lab(rgb): return xyz_to_lab(rgb_to_xyz(rgb))
def deltaE76(c1,c2):
    L1,a1,b1 = rgb_to_lab(c1); L2,a2,b2 = rgb_to_lab(c2)
    return math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)

# WCAG-like contrast
def relative_luminance(rgb):
    r,g,b = to01(rgb)
    rl,gl,bl = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)
    return 0.2126*rl + 0.7152*gl + 0.0722*bl
def contrast_ratio(a,b):
    L1,L2 = relative_luminance(a), relative_luminance(b)
    hi,lo = max(L1,L2), min(L1,L2)
    return (hi + 0.05) / (lo + 0.05)

# -----------------------------
# CVD transforms
# -----------------------------
DEFAULT_CVD_MATS = {
    "protanopia": np.array([[0.56667,0.43333,0],[0.55833,0.44167,0],[0,0.24167,0.75833]]),
    "deuteranopia": np.array([[0.625,0.375,0],[0.7,0.3,0],[0,0.3,0.7]]),
    "tritanopia": np.array([[0.95,0.05,0],[0,0.43333,0.56667],[0,0.475,0.525]]),
    "normal": np.eye(3),
}
def apply_matrix_to_rgb(rgb, mat):
    r,g,b = to01(rgb)
    mat = np.array(mat)
    vec = np.array([r,g,b])
    out = mat.dot(vec)
    return to255(tuple(clamp01(x) for x in out))

def build_cvd_matrix(cvd_profile: Dict[str,Any]) -> np.ndarray:
    if not cvd_profile: return DEFAULT_CVD_MATS["normal"]
    if "transform_matrix" in cvd_profile and cvd_profile["transform_matrix"] is not None:
        M = np.array(cvd_profile["transform_matrix"], dtype=float)
        if M.shape==(3,3): return M
    typ = cvd_profile.get("type","normal")
    sev = float(cvd_profile.get("severity", 0.0))
    base = DEFAULT_CVD_MATS.get(typ, DEFAULT_CVD_MATS["normal"])
    M = sev*base + (1.0-sev)*np.eye(3)
    cs = cvd_profile.get("cone_sensitivities")
    if cs:
        row_mul = np.array([cs.get('L',1.0), cs.get('M',1.0), cs.get('S',1.0)])
        M = (M.T * row_mul).T
    return M

# -----------------------------
# Enhanced Fashion Analysis Functions
# -----------------------------

def _try_import_skfuzzy():
    try:
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        return fuzz, ctrl
    except Exception:
        return None, None

def get_membership(fuzzy_values, var_range, var_model, crisp_value):
    """Returns the fuzzy membership name with highest membership value"""
    fuzz, _ = _try_import_skfuzzy()
    if not fuzz:
        return fuzzy_values[0]
    
    max_membership = 0
    membership_name = fuzzy_values[0]
    for i in range(len(fuzzy_values)):
        temp_memb = fuzz.interp_membership(var_range, var_model[fuzzy_values[i]].mf, crisp_value)
        if temp_memb > max_membership:
            max_membership = temp_memb
            membership_name = fuzzy_values[i]
    return membership_name

def build_tone_system():
    """Build fuzzy system for tone analysis (NEUTRAL, DARK, BRIGHT)"""
    fuzz, ctrl = _try_import_skfuzzy()
    if not fuzz:
        return None, None, None
    
    # Inputs: saturation and value
    sat = ctrl.Antecedent(np.arange(0, 101, 1), 'saturation')
    val = ctrl.Antecedent(np.arange(0, 101, 1), 'value')
    
    # Saturation memberships
    sat['GRAY'] = fuzz.gaussmf(sat.universe, 0, 10)
    sat['VERY_FADED'] = fuzz.gaussmf(sat.universe, 25, 10)
    sat['FADED'] = fuzz.gaussmf(sat.universe, 50, 10)
    sat['SATURATED'] = fuzz.gaussmf(sat.universe, 75, 10)
    sat['VERY_SATURATED'] = fuzz.gaussmf(sat.universe, 100, 10)
    
    # Value memberships
    val['BLACK'] = fuzz.gaussmf(val.universe, 0, 10)
    val['VERY_DARK'] = fuzz.gaussmf(val.universe, 25, 10)
    val['DARK'] = fuzz.gaussmf(val.universe, 50, 10)
    val['BRIGHT'] = fuzz.gaussmf(val.universe, 75, 10)
    val['VERY_BRIGHT'] = fuzz.gaussmf(val.universe, 100, 10)
    
    # Output: tone
    tone_range = np.arange(0, 12, 1)
    tone = ctrl.Consequent(tone_range, 'tone')
    tone['NEUTRAL'] = fuzz.trapmf(tone.universe, [0, 0, 1, 2])
    tone['DARK'] = fuzz.gbellmf(tone.universe, 2, 1, 3)
    tone['BRIGHT'] = fuzz.gbellmf(tone.universe, 4, 1, 9.5)
    
    # Rules for tone determination
    rules = [
        ctrl.Rule(val['BLACK'] | sat['GRAY'] | sat['VERY_FADED'], tone['NEUTRAL']),
        ctrl.Rule(val['VERY_DARK'] & sat['SATURATED'], tone['NEUTRAL']),
        ctrl.Rule(val['DARK'] & sat['FADED'], tone['DARK']),
        ctrl.Rule(val['DARK'] & sat['VERY_SATURATED'], tone['BRIGHT']),
        ctrl.Rule(val['BRIGHT'] & sat['SATURATED'], tone['BRIGHT']),
        ctrl.Rule(val['VERY_BRIGHT'] & sat['FADED'], tone['BRIGHT']),
        ctrl.Rule(val['VERY_BRIGHT'] & sat['VERY_SATURATED'], tone['BRIGHT']),
        ctrl.Rule(val['VERY_DARK'] & sat['FADED'], tone['NEUTRAL'])
    ]
    
    tone_ctrl = ctrl.ControlSystem(rules)
    tone_sim = ctrl.ControlSystemSimulation(tone_ctrl)
    
    return tone_sim, tone_range, tone

def build_temperature_system():
    """Build fuzzy system for color temperature (WARM, COOL)"""
    fuzz, ctrl = _try_import_skfuzzy()
    if not fuzz:
        return None, None, None
    
    hue_range = np.arange(0, 361, 1)
    hue = ctrl.Antecedent(hue_range, 'hue')
    
    # Temperature memberships - WARM colors around red/yellow, COOL around blue/cyan
    hue['WARM'] = fuzz.gaussmf(hue.universe, 0, 60)      # Red region
    hue['COOL'] = fuzz.gaussmf(hue.universe, 180, 60)    # Cyan/Blue region  
    hue['WARM_'] = fuzz.gaussmf(hue.universe, 360, 60)   # Red region (wraparound)
    
    return None, hue_range, hue  # No rules needed, just membership

def get_color_tone(sat_val, val_val, tone_system=None):
    """Get tone classification (NEUTRAL, DARK, BRIGHT) for given saturation and value"""
    if tone_system is None:
        tone_sim, tone_range, tone = build_tone_system()
        if tone_sim is None:
            # Fallback logic
            if val_val < 25 or sat_val < 20:
                return 'NEUTRAL'
            elif val_val < 60:
                return 'DARK'
            else:
                return 'BRIGHT'
    else:
        tone_sim, tone_range, tone = tone_system
    
    tone_sim.input['saturation'] = max(0, min(100, sat_val))
    tone_sim.input['value'] = max(0, min(100, val_val))
    tone_sim.compute()
    tone_output = tone_sim.output['tone']
    
    tone_fuzzy = ['NEUTRAL', 'DARK', 'BRIGHT']
    return get_membership(tone_fuzzy, tone_range, tone, tone_output)

def get_color_temperature(hue_val, temp_system=None):
    """Get temperature classification (WARM, COOL) for given hue"""
    if temp_system is None:
        _, hue_range, hue = build_temperature_system()
        if hue is None:
            # Fallback logic
            if (hue_val >= 0 and hue_val <= 60) or hue_val >= 300:
                return 'WARM'
            else:
                return 'COOL'
    else:
        _, hue_range, hue = temp_system
    
    hue_fuzzy = ['WARM', 'COOL', 'WARM_']
    temp = get_membership(hue_fuzzy, hue_range, hue, hue_val)
    return 'WARM' if temp == 'WARM_' else temp

def get_color_description(rgb):
    """Get complete color description (tone, temperature) for RGB color"""
    h, s, v = rgb_to_hsv_deg(rgb)
    tone = get_color_tone(s, v)
    temp = get_color_temperature(h)
    return (tone, temp)

# -----------------------------
# Fashion Outfit Matching Rules
# -----------------------------

def basic_match(outfit_colors):
    """
    Basic outfit rules:
    - No more than one bright color
    - Any number of neutral colors
    """
    descriptions = [get_color_description(color) for color in outfit_colors]
    bright_count = len([desc for desc in descriptions if desc[0] == 'BRIGHT'])
    return bright_count <= 1

def neutral_match(outfit_colors):
    """All colors must be neutral"""
    descriptions = [get_color_description(color) for color in outfit_colors]
    neutral_count = len([desc for desc in descriptions if desc[0] == 'NEUTRAL'])
    return neutral_count == len(descriptions)

def analogous_match(outfit_colors):
    """All non-neutral colors must be within the same temperature"""
    descriptions = [get_color_description(color) for color in outfit_colors]
    non_neutral = [desc for desc in descriptions if desc[0] != 'NEUTRAL']
    
    if len(non_neutral) <= 1:
        return True
    
    cool_count = len([desc for desc in non_neutral if desc[1] == 'COOL'])
    warm_count = len(non_neutral) - cool_count
    
    return cool_count == 0 or warm_count == 0

def contrast_match(outfit_colors):
    """
    Contrast outfit rules:
    - At least one warm color
    - Both dark and bright colors present
    """
    descriptions = [get_color_description(color) for color in outfit_colors]
    non_neutral = [desc for desc in descriptions if desc[0] != 'NEUTRAL']
    
    warm_count = len([desc for desc in descriptions if desc[1] == 'WARM'])
    if warm_count < 1:
        return False
    
    dark_count = len([desc for desc in non_neutral if desc[0] == 'DARK'])
    bright_count = len([desc for desc in non_neutral if desc[0] == 'BRIGHT'])
    
    return dark_count >= 1 and bright_count >= 1

def summer_match(outfit_colors):
    """
    Summer outfit rules:
    - At least two warm colors (non-neutral)
    - At least one bright color
    - At most one dark color
    """
    descriptions = [get_color_description(color) for color in outfit_colors]
    non_neutral = [desc for desc in descriptions if desc[0] != 'NEUTRAL']
    
    warm_count = len([desc for desc in non_neutral if desc[1] == 'WARM'])
    if warm_count < 2:
        return False
    
    dark_count = len([desc for desc in non_neutral if desc[0] == 'DARK'])
    if dark_count > 1:
        return False
    
    bright_count = len(non_neutral) - dark_count
    return bright_count >= 1

def winter_match(outfit_colors):
    """
    Winter outfit rules:
    - At least one dark color
    - No bright colors (among non-neutral)
    """
    descriptions = [get_color_description(color) for color in outfit_colors]
    non_neutral = [desc for desc in descriptions if desc[0] != 'NEUTRAL']
    
    dark_count = len([desc for desc in non_neutral if desc[0] == 'DARK'])
    if dark_count < 1:
        return False
    
    bright_count = len(non_neutral) - dark_count
    return bright_count == 0

def get_valid_matches(outfit_colors):
    """Get all valid outfit matching schemes for given colors"""
    rules = {
        "Basic": basic_match,
        "Neutral": neutral_match,
        "Analogous": analogous_match,
        "Contrast": contrast_match,
        "Summer": summer_match,
        "Winter": winter_match
    }
    
    valid_matches = []
    for scheme_name, rule_func in rules.items():
        if rule_func(outfit_colors):
            valid_matches.append(scheme_name)
    
    return valid_matches

# -----------------------------
# Enhanced Fashion Scoring
# -----------------------------

def circular_diff_deg(a,b):
    d = abs((a-b)%360.0)
    return d if d<=180 else 360-d

def calculate_fashion_score(candidate_rgb, primary_colors, context=None):
    """
    Calculate fashion compatibility score using improved fuzzy rules
    Returns score 0-100
    """
    context = context or {}
    outfit_colors = primary_colors + [candidate_rgb]
    
    # Get valid matching schemes
    valid_schemes = get_valid_matches(outfit_colors)
    
    # Base score from number of valid schemes
    base_score = len(valid_schemes) * 15.0  # Max ~90 for 6 schemes
    
    # Analyze color relationships
    candidate_desc = get_color_description(candidate_rgb)
    primary_descs = [get_color_description(color) for color in primary_colors]
    
    # Bonus for complementary relationships
    h_cand, s_cand, v_cand = rgb_to_hsv_deg(candidate_rgb)
    harmony_bonus = 0.0
    
    for primary in primary_colors:
        h_prim, s_prim, v_prim = rgb_to_hsv_deg(primary)
        hue_diff = circular_diff_deg(h_cand, h_prim)
        
        # Reward complementary (150-210°), triadic (110-130°), and analogous (20-40°)
        if 150 <= hue_diff <= 210:  # Complementary
            harmony_bonus += 15.0
        elif 110 <= hue_diff <= 130:  # Triadic
            harmony_bonus += 12.0
        elif 20 <= hue_diff <= 40:  # Analogous
            harmony_bonus += 8.0
    
    # Context-based adjustments
    formality = context.get("formality", 50)
    season = context.get("season", "spring")
    
    # Formality adjustments
    if formality > 70:  # Formal
        if "Basic" in valid_schemes or "Neutral" in valid_schemes:
            base_score += 10
        if "Contrast" in valid_schemes:
            base_score -= 5
    elif formality < 30:  # Casual
        if "Summer" in valid_schemes or "Contrast" in valid_schemes:
            base_score += 8
    
    # Season adjustments
    if season.lower() == "summer" and "Summer" in valid_schemes:
        base_score += 10
    elif season.lower() == "winter" and "Winter" in valid_schemes:
        base_score += 10
    
    final_score = min(100.0, max(0.0, base_score + harmony_bonus))
    
    return {
        "score": final_score,
        "valid_schemes": valid_schemes,
        "candidate_desc": candidate_desc,
        "harmony_bonus": harmony_bonus,
        "base_score": base_score
    }

# -----------------------------
# Candidate generation
# -----------------------------

def generate_theory_candidates(rgb):
    h,s,v = rgb_to_hsv_deg(rgb)
    targets = [h+180, h+120, h-120, h+30, h-30, h+150, h-150]
    sv_vars = [(s,v),(min(100,s*1.05),v),(s,min(100,v*1.05)),(max(6,s*0.85),v)]
    pool=[]
    for th in targets:
        for (ts,tv) in sv_vars:
            pool.append(hsv_deg_to_rgb(th, ts, tv))
    # ΔE dedupe
    uniq=[]
    for c in pool:
        if all(deltaE76(c,u)>4.5 for u in uniq):
            uniq.append(c)
    return uniq

def expand_color_inputs(color_list):
    out=[]
    for item in (color_list or []):
        if isinstance(item,(list,tuple)) and len(item)==2 and isinstance(item[1],(int,float)):
            col,prop = item[0], float(item[1])
            rgb = hex_to_rgb(col) if isinstance(col,str) else tuple(col)
            out.append((rgb,prop))
        else:
            rgb = hex_to_rgb(item) if isinstance(item,str) else tuple(item)
            out.append((rgb,1.0))
    tot = sum(p for _,p in out) or 1.0
    return [(c,p/tot) for c,p in out]

def aggregate_contrast_metrics(candidate, refs):
    if not refs:
        return {"cr_mean":0.0, "cr_min":0.0, "de_mean":0.0, "de_min":0.0}
    crs = [contrast_ratio(candidate, r) for r in refs]
    des = [deltaE76(candidate, r) for r in refs]
    return {"cr_mean":sum(crs)/len(crs), "cr_min":min(crs), "de_mean":sum(des)/len(des), "de_min":min(des)}

# -----------------------------
# Enhanced Visualization & Explanation Functions
# -----------------------------

def create_outfit_visualization(outfit_colors, outfit_labels=None, title="Outfit Color Scheme", 
                               valid_schemes=None, save_path=None, show_cvd_comparison=False, cvd_matrix=None):
    """
    Create a visual representation of the outfit similar to your example image
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("Matplotlib not available for visualization")
        return None
    
    if outfit_labels is None:
        outfit_labels = [f"Color {i+1}" for i in range(len(outfit_colors))]
    
    # Convert colors to RGB format for matplotlib
    rgb_colors = []
    for color in outfit_colors:
        if isinstance(color, str):
            rgb = hex_to_rgb(color)
        else:
            rgb = color
        rgb_colors.append([c/255.0 for c in rgb])
    
    # Create figure
    fig_width = 8 if not show_cvd_comparison else 12
    fig, axes = plt.subplots(1, 2 if show_cvd_comparison else 1, figsize=(fig_width, 6))
    if not show_cvd_comparison:
        axes = [axes]
    
    for ax_idx, ax in enumerate(axes):
        # Use CVD-transformed colors for second subplot
        if ax_idx == 1 and show_cvd_comparison and cvd_matrix is not None:
            display_colors = []
            for color in outfit_colors:
                if isinstance(color, str):
                    rgb = hex_to_rgb(color)
                else:
                    rgb = color
                cvd_rgb = apply_matrix_to_rgb(rgb, cvd_matrix)
                display_colors.append([c/255.0 for c in cvd_rgb])
        else:
            display_colors = rgb_colors
        
        # Create color bars
        bar_height = 0.8
        for i, (color, label) in enumerate(zip(display_colors, outfit_labels)):
            y_pos = len(outfit_colors) - i - 1
            rect = patches.Rectangle((0, y_pos), 1, bar_height, 
                                   linewidth=1, edgecolor='black', facecolor=color)
            ax.add_patch(rect)
            ax.text(1.05, y_pos + bar_height/2, label.upper(), 
                   va='center', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.1, len(outfit_colors))
        ax.set_aspect('equal')
        ax.axis('off')
        
        subplot_title = title
        if show_cvd_comparison:
            subplot_title += " (CVD View)" if ax_idx == 1 else " (Normal View)"
        ax.set_title(subplot_title, fontsize=14, fontweight='bold', pad=20)
    
    # Add scheme information
    if valid_schemes:
        scheme_text = f"Valid outfit schemes: {valid_schemes}"
        fig.text(0.5, 0.02, scheme_text, ha='center', fontsize=11, 
                style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    return fig

def generate_detailed_explanation(candidate_rgb, primary_colors, fashion_analysis, cvd_metrics, context=None):
    """
    Generate comprehensive fuzzy explanation of color relationships
    """
    context = context or {}
    candidate_desc = fashion_analysis["candidate_desc"]
    candidate_hex = rgb_to_hex(candidate_rgb)
    
    explanation = []
    explanation.append(f"RECOMMENDED COLOR: {candidate_hex}")
    explanation.append(f"Color Classification: {candidate_desc[0]} {candidate_desc[1]}")
    
    # HSV breakdown
    h, s, v = rgb_to_hsv_deg(candidate_rgb)
    explanation.append(f"HSV Values: Hue={h:.0f}°, Saturation={s:.0f}%, Value={v:.0f}%")
    
    # Color relationships with primaries
    explanation.append("\nCOLOR HARMONY ANALYSIS:")
    for i, primary in enumerate(primary_colors):
        p_hex = rgb_to_hex(primary)
        p_desc = get_color_description(primary)
        ph, ps, pv = rgb_to_hsv_deg(primary)
        
        hue_diff = circular_diff_deg(h, ph)
        sat_diff = abs(s - ps)
        val_diff = abs(v - pv)
        
        # Determine harmony type
        harmony_type = "Monochromatic"
        if 150 <= hue_diff <= 210:
            harmony_type = "Complementary"
        elif 110 <= hue_diff <= 130:
            harmony_type = "Triadic"
        elif 20 <= hue_diff <= 40:
            harmony_type = "Analogous"
        elif 60 <= hue_diff <= 80:
            harmony_type = "Split-Complementary"
        
        explanation.append(f"  vs Primary {i+1} ({p_hex} - {p_desc[0]} {p_desc[1]}):")
        explanation.append(f"    Hue difference: {hue_diff:.0f}° → {harmony_type}")
        explanation.append(f"    Saturation contrast: {sat_diff:.0f}% {'(High)' if sat_diff > 40 else '(Moderate)' if sat_diff > 15 else '(Low)'}")
        explanation.append(f"    Value contrast: {val_diff:.0f}% {'(High)' if val_diff > 35 else '(Moderate)' if val_diff > 15 else '(Low)'}")
    
    # Outfit scheme analysis
    explanation.append(f"\nOUTFIT SCHEME COMPATIBILITY:")
    valid_schemes = fashion_analysis["valid_schemes"]
    if valid_schemes:
        explanation.append(f"✓ Matches {len(valid_schemes)} scheme(s): {', '.join(valid_schemes)}")
        
        # Explain why each scheme works
        for scheme in valid_schemes:
            if scheme == "Basic":
                explanation.append("  • Basic: Safe color combination with balanced contrast")
            elif scheme == "Neutral":
                explanation.append("  • Neutral: All colors are neutral tones - very versatile")
            elif scheme == "Analogous":
                explanation.append("  • Analogous: Colors share same temperature family - harmonious")
            elif scheme == "Contrast":
                explanation.append("  • Contrast: Bold mix of warm and cool with varied brightness")
            elif scheme == "Summer":
                explanation.append("  • Summer: Bright, warm palette perfect for casual sunny days")
            elif scheme == "Winter":
                explanation.append("  • Winter: Sophisticated dark palette with no bright distractions")
    else:
        explanation.append("⚠ No standard outfit schemes matched")
    
    # Context appropriateness
    formality = context.get("formality", 50)
    if formality > 70:
        explanation.append(f"\nFORMAL CONTEXT (Level: {formality}):")
        if "Basic" in valid_schemes or "Neutral" in valid_schemes:
            explanation.append("✓ Excellent for formal settings - conservative and professional")
        elif "Contrast" in valid_schemes:
            explanation.append("⚠ May be too bold for very formal occasions")
        else:
            explanation.append("○ Moderate formality - suitable for business casual")
    elif formality < 30:
        explanation.append(f"\nCASUAL CONTEXT (Level: {formality}):")
        if "Summer" in valid_schemes or "Contrast" in valid_schemes:
            explanation.append("✓ Perfect for casual wear - expressive and comfortable")
        else:
            explanation.append("○ Works for casual but may be understated")
    
    # CVD accessibility
    explanation.append(f"\nCVD ACCESSIBILITY:")
    explanation.append(f"Color Difference (ΔE): {cvd_metrics['de_min']:.1f} (need ≥10 for safety)")
    explanation.append(f"Contrast Ratio: {cvd_metrics['cr_min']:.2f} (need ≥1.2 for safety)")
    if cvd_metrics["de_min"] >= 10 and cvd_metrics["cr_min"] >= 1.2:
        explanation.append("✓ CVD-SAFE: Colors remain distinguishable with color vision differences")
    else:
        explanation.append("⚠ CVD-RISK: May be difficult to distinguish with color vision differences")
    
    return "\n".join(explanation)

def display_comprehensive_recommendation(candidate_info, primary_colors, cvd_profile=None, context=None):
    """
    Display complete recommendation with visualization and detailed explanation
    """
    candidate_rgb = candidate_info["rgb"]
    fashion_analysis = candidate_info["fashion_analysis"]
    cvd_metrics = candidate_info["cvd_metrics_primary"]
    cvd_matrix = np.array(candidate_info["cvd_mat"]) if "cvd_mat" in candidate_info else None
    
    # Create outfit visualization
    outfit_colors = primary_colors + [candidate_rgb]
    outfit_labels = [f"Primary {i+1}" for i in range(len(primary_colors))] + ["Recommended"]
    valid_schemes_str = f"[{', '.join([f'\'{s}\'' for s in fashion_analysis['valid_schemes']])}]"
    
    fig = create_outfit_visualization(
        outfit_colors, 
        outfit_labels, 
        title="Outfit Color Scheme",
        valid_schemes=valid_schemes_str,
        show_cvd_comparison=bool(cvd_profile and cvd_profile.get("type") != "normal"),
        cvd_matrix=cvd_matrix
    )
    
    if fig:
        try:
            plt.show()
        except:
            pass
    
    # Generate detailed explanation
    explanation = generate_detailed_explanation(
        candidate_rgb, primary_colors, fashion_analysis, cvd_metrics, context
    )
    
    print("="*80)
    print("COMPREHENSIVE COLOR RECOMMENDATION ANALYSIS")
    print("="*80)
    print(explanation)
    print("="*80)
    
    return fig, explanation

# -----------------------------
# Enhanced recommendation function
# -----------------------------

def recommend(
    primaries: List[Any],
    secondaries: Optional[List[Any]] = None,
    cvd_profile: Optional[Dict[str,Any]] = None,
    topk: int = 5,
    context: Optional[Dict[str,Any]] = None,
    feedback_log: Optional[str] = "reco_feedback.csv"
) -> List[Dict[str,Any]]:
    """
    Enhanced recommendation with improved fashion analysis
    """
    context = context or {}
    
    # Parse inputs w/ proportions
    prims = expand_color_inputs(primaries)
    secs  = expand_color_inputs(secondaries or [])
    prim_colors = [c for c,_ in prims]
    sec_colors  = [c for c,_ in secs]

    # Build personalized CVD matrix
    cvd_profile = cvd_profile or {"type":"normal","severity":0.0}
    cvd_mat = build_cvd_matrix(cvd_profile)
    min_de = float(cvd_profile.get("min_cvd_deltaE", 10.0))
    min_cr = float(cvd_profile.get("min_cvd_contrast", 1.2))

    # Candidate pool from color theory
    pool=[]
    for c in prim_colors:
        pool += generate_theory_candidates(c)
    if len(prim_colors)>=2:
        for a,b in itertools.combinations(prim_colors,2):
            ha,sa,va = rgb_to_hsv_deg(a); hb,sb,vb = rgb_to_hsv_deg(b)
            hmid=(ha+hb)/2.0; smid=(sa+sb)/2.0; vmid=(va+vb)/2.0
            pool += generate_theory_candidates(hsv_deg_to_rgb(hmid, smid, vmid))

    # Deduplicate
    uniq=[]
    for c in pool:
        if all(deltaE76(c,u)>5.0 for u in uniq):
            uniq.append(c)
    pool = [c for c in uniq if all(deltaE76(c,e)>8.0 for e in (prim_colors+sec_colors))] or uniq

    scored=[]
    for cand in pool:
        # Enhanced fashion analysis
        fashion_analysis = calculate_fashion_score(cand, prim_colors, context)
        fashion_score = fashion_analysis["score"]
        
        # Simulate CVD
        cand_sim = apply_matrix_to_rgb(cand, cvd_mat)
        prims_sim = [apply_matrix_to_rgb(p, cvd_mat) for p in prim_colors]
        secs_sim  = [apply_matrix_to_rgb(s, cvd_mat) for s in sec_colors]

        # CVD distinguishability
        cb_prim = aggregate_contrast_metrics(cand_sim, prims_sim)
        cb_sec  = aggregate_contrast_metrics(cand_sim, secs_sim) if secs_sim else {"cr_mean":0,"cr_min":0,"de_mean":0,"de_min":0}
        cb_norm = aggregate_contrast_metrics(cand, prim_colors+sec_colors)

        # Enhanced composite score
        score = (
            0.60 * (fashion_score/100.0) +           # Primary: fashion compatibility
            0.30 * (0.6*(cb_prim["de_mean"]/50.0) + 0.4*(cb_prim["cr_mean"]/7.0)) +  # CVD distinguishability
            0.10 * (0.5*(cb_norm["de_mean"]/50.0) + 0.5*(cb_norm["cr_mean"]/7.0))    # Normal contrast
        )

        # CVD safety gate
        cvd_safe = (cb_prim["de_min"] >= min_de) and (cb_prim["cr_min"] >= min_cr)

        scored.append({
            "hex": rgb_to_hex(cand),
            "rgb": cand,
            "score": float(score),
            "cvd_safe": bool(cvd_safe),
            "fashion_score": float(fashion_score),
            "fashion_analysis": fashion_analysis,
            "cvd_mat": cvd_mat.tolist(),
            "cvd_metrics_primary": cb_prim,
            "cvd_metrics_secondary": cb_sec,
            "normal_metrics": cb_norm,
            "explanation": (
                f"Fashion score={fashion_score:.0f}/100 "
                f"(schemes: {', '.join(fashion_analysis['valid_schemes']) or 'None'}); "
                f"CVD ΔE(min/mean)={cb_prim['de_min']:.1f}/{cb_prim['de_mean']:.1f}, "
                f"CR(min/mean)={cb_prim['cr_min']:.2f}/{cb_prim['cr_mean']:.2f}; "
                f"Color: {fashion_analysis['candidate_desc'][0]} {fashion_analysis['candidate_desc'][1]}."
            )
        })

    # Rank: CVD-safe first, then by score
    safe = [s for s in scored if s["cvd_safe"]]
    ranked = sorted(safe or scored, key=lambda d: d["score"], reverse=True)
    top = ranked[:topk]

    # Optional feedback logging
    if feedback_log:
        try:
            newfile = not os.path.exists(feedback_log)
            with open(feedback_log, "a", newline='', encoding="utf-8") as f:
                w = csv.writer(f)
                if newfile:
                    w.writerow(["primaries","secondaries","cvd_profile","context","candidate_hex","score","cvd_safe","fashion_score","schemes"])
                for r in top:
                    w.writerow([
                        json.dumps([rgb_to_hex(c) for c in prim_colors]),
                        json.dumps([rgb_to_hex(c) for c in sec_colors]),
                        json.dumps(cvd_profile),
                        json.dumps(context),
                        r["hex"], f"{r['score']:.4f}", int(r["cvd_safe"]), 
                        int(round(r["fashion_score"])), 
                        "|".join(r["fashion_analysis"]["valid_schemes"])
                    ])
        except Exception as e:
            print("Feedback log failed:", e)

    return top

# -----------------------------
# Enhanced Demo with Comprehensive Output
# -----------------------------
if __name__ == "__main__":
    print("="*60)
    print("ENHANCED FUZZY COLOR RECOMMENDER WITH VISUAL OUTPUT")
    print("="*60)
    
    # Test the enhanced system with comprehensive output
    prim = [("#1F3A93", 0.7), ("#FFFFFF", 0.3)]   # navy + white
    sec  = ["#D2B48C"]                            # tan
    cvd_profile = {
        "type":"deuteranopia",
        "severity":0.8,
        "min_cvd_deltaE": 12.0,
        "min_cvd_contrast": 1.25,
    }
    ctx = {"formality": 75, "season": "winter"}  # Formal winter outfit
    
    print("Input Configuration:")
    print(f"  Primary Colors: {[rgb_to_hex(hex_to_rgb(p[0])) for p in prim]}")
    print(f"  Secondary Colors: {sec}")
    print(f"  CVD Profile: {cvd_profile['type']} (severity: {cvd_profile['severity']})")
    print(f"  Context: Formality={ctx['formality']}, Season={ctx['season']}")
    print("\nGenerating recommendations...")
    
    recs = recommend(prim, sec, cvd_profile=cvd_profile, topk=3, context=ctx)
    
    # Parse primary colors for visualization
    prims = expand_color_inputs(prim)
    prim_colors = [c for c,_ in prims]
    
    # Display each recommendation with comprehensive analysis
    for i, r in enumerate(recs, 1):
        print(f"\n{'='*20} RECOMMENDATION #{i} {'='*20}")
        print(f"Color: {r['hex']} | Overall Score: {r['score']:.3f} | Fashion Score: {r['fashion_score']:.0f}/100")
        print(f"CVD Safe: {'✓' if r['cvd_safe'] else '✗'}")
        
        # Display comprehensive analysis
        display_comprehensive_recommendation(r, prim_colors, cvd_profile, ctx)
        
        if i < len(recs):
            input("\nPress Enter for next recommendation...")
    
    # print("\n" + "="*60)
    # print("SYSTEM CAPABILITIES DEMONSTRATED:")
    # print("="*60)
    # print("✓ HSV-based fuzzy color analysis")
    # print("✓ Multiple outfit scheme validation")
    # print("✓ Visual outfit representation")
    # print("✓ CVD simulation and comparison")
    # print("✓ Detailed harmony analysis")
    # print("✓ Context-aware recommendations")
    # print("✓ Comprehensive explanations")
    # print("✓ Professional fashion scoring")
    
    # # Quick test of individual features
    # print("\n" + "="*40)
    # print("QUICK FEATURE TEST:")
    # print("="*40)
    
    # test_colors = [
    #     ("#FF0000", "Pure Red"),
    #     ("#87CEEB", "Sky Blue"), 
    #     ("#2F4F4F", "Dark Slate Gray"),
    #     ("#FFD700", "Gold")
    # ]
    
    # print("Color Classifications:")
    # for hex_color, name in test_colors:
    #     rgb = hex_to_rgb(hex_color)
    #     desc = get_color_description(rgb)
    #     h, s, v = rgb_to_hsv_deg(rgb)
    #     print(f"  {name}: {desc[0]} {desc[1]} - HSV({h:.0f}, {s:.0f}, {v:.0f})")
    
    # # Test outfit validation
    # print("\nOutfit Scheme Validation:")
    # test_outfit = [hex_to_rgb("#1F3A93"), hex_to_rgb("#FFFFFF"), hex_to_rgb("#D2B48C")]
    # schemes = get_valid_matches(test_outfit)
    # print(f"  Navy + White + Tan: {', '.join(schemes) or 'No valid schemes'}")
    
    # print("\nSystem ready for production use!")