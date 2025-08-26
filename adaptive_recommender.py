# adaptive_recommender.py
# Run: python adaptive_recommender.py   (demo at bottom)
import math, json, os, csv, itertools
from typing import List, Tuple, Dict, Optional, Any
import colorsys
import numpy as np
from copy import deepcopy

# --- utilities: color conversions and metrics (sRGB/D65) ---
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

# --- XYZ/Lab conversion for ΔE76 ---
def srgb_to_linear(c): return c/12.92 if c<=0.04045 else ((c+0.055)/1.055)**2.4
def linear_to_srgb(c): return 12.92*c if c<=0.0031308 else 1.055*(c**(1/2.4))-0.055
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
    fx, fy, fz = f(X/Xr), f(Y/Yr), f(Z/Zr)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return (L,a,b)
def rgb_to_lab(rgb): return xyz_to_lab(rgb_to_xyz(rgb))
def deltaE76(c1,c2):
    L1,a1,b1 = rgb_to_lab(c1); L2,a2,b2 = rgb_to_lab(c2)
    return math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)

# --- WCAG-like contrast ---
def relative_luminance(rgb):
    r,g,b = to01(rgb)
    rl,gl,bl = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)
    return 0.2126*rl + 0.7152*gl + 0.0722*bl
def contrast_ratio(a,b):
    L1,L2 = relative_luminance(a), relative_luminance(b)
    hi,lo = max(L1,L2), min(L1,L2)
    return (hi + 0.05) / (lo + 0.05)

# --- default CVD matrices (fallback) ---
DEFAULT_CVD_MATS = {
    "protanopia": np.array([[0.56667,0.43333,0],[0.55833,0.44167,0],[0,0.24167,0.75833]]),
    "deuteranopia": np.array([[0.625,0.375,0],[0.7,0.3,0],[0,0.3,0.7]]),
    "tritanopia": np.array([[0.95,0.05,0],[0,0.43333,0.56667],[0,0.475,0.525]]),
    "normal": np.eye(3)
}

def apply_matrix_to_rgb(rgb, mat):
    r,g,b = to01(rgb)
    vec = np.array([r,g,b])
    out = mat.dot(vec)
    return to255(tuple(clamp01(x) for x in out))

# --- build CVD transform from profile ---
def build_cvd_matrix_from_profile(cvd_profile: Dict[str,Any]) -> np.ndarray:
    # Prefers explicit transform_matrix; otherwise constructs from severity + cone sensitivities
    if not cvd_profile:
        return DEFAULT_CVD_MATS["normal"]
    if "transform_matrix" in cvd_profile and cvd_profile["transform_matrix"] is not None:
        mat = np.array(cvd_profile["transform_matrix"], dtype=float)
        if mat.shape == (3,3):
            return mat
    # fallback: use default base mat per type and interpolate to normal via severity
    typ = cvd_profile.get("type","normal")
    severity = float(cvd_profile.get("severity",0.9))  # default high if provided
    base = DEFAULT_CVD_MATS.get(typ, DEFAULT_CVD_MATS["normal"])
    # linearly blend with identity by severity: severity=1 -> base, severity=0->identity
    mat = severity * base + (1.0 - severity) * np.eye(3)
    # apply cone_sensitivities if provided (scale rows)
    cs = cvd_profile.get("cone_sensitivities")
    if cs:
        # cs expected to map 'L','M','S' to scalar multipliers
        # roughly map L->R row, M->G row, S->B row
        row_mul = np.array([cs.get('L',1.0), cs.get('M',1.0), cs.get('S',1.0)])
        mat = (mat.T * row_mul).T
    return mat

# --- Candidate generation (color theory variants) ---
def circular_diff_deg(a,b):
    d = abs((a-b) % 360.0)
    return d if d<=180 else 360-d

def harmony_kernel(angle_diff, center, sigma=24.0):
    return math.exp(-0.5 * ((angle_diff - center)/sigma)**2)

def generate_theory_candidates(rgb):
    h,s,v = rgb_to_hsv_deg(rgb)
    targets = [h+180, h+120, h-120, h+30, h-30, h+150, h-150]
    sv_vars = [(s,v), (min(100,s*1.05),v), (s,min(100,v*1.05)), (max(6,s*0.85), v)]
    pool = []
    for th in targets:
        for (ts,tv) in sv_vars:
            pool.append(hsv_deg_to_rgb(th, ts, tv))
    # remove duplicates by ΔE small threshold
    uniq = []
    for c in pool:
        if all(deltaE76(c,u) > 4.5 for u in uniq):
            uniq.append(c)
    return uniq

# --- aggregate primaries/secondaries (handle proportions) ---
def expand_color_inputs(color_list):
    # color_list can be: list of hex / rgb / or tuples (rgb, proportion)
    out = []
    for item in color_list:
        if isinstance(item, (list,tuple)) and len(item)==2 and isinstance(item[1], (int,float)):
            # assume (rgb_or_hex, proportion)
            col, prop = item[0], float(item[1])
            rgb = hex_to_rgb(col) if isinstance(col,str) else tuple(col)
            out.append( (rgb, prop) )
        else:
            rgb = hex_to_rgb(item) if isinstance(item,str) else tuple(item)
            out.append( (rgb, 1.0) )
    # normalize proportions
    total = sum(p for _,p in out) or 1.0
    out = [(c, p/total) for c,p in out]
    return out

# --- scoring functions ---
def harmony_score_for_candidate(candidate, primaries):
    # measure average best-rule response across primaries
    scores = []
    ch = rgb_to_hsv_deg(candidate)[0]
    for p in primaries:
        ph = rgb_to_hsv_deg(p)[0]
        d = circular_diff_deg(ch, ph)
        comp = harmony_kernel(d, 180, 20)
        tri  = harmony_kernel(d, 120, 18)
        ana  = harmony_kernel(d, 30, 12)
        split = max(harmony_kernel(d,150,18), harmony_kernel(d,210,18))
        scores.append(max(comp, tri, ana, split))
    return sum(scores)/len(scores) if scores else 0.0

def aggregate_contrast_metrics(candidate, refs):
    if not refs:
        return {"cr_mean":0.0, "cr_min":0.0, "de_mean":0.0, "de_min":0.0}
    crs = [contrast_ratio(candidate, r) for r in refs]
    des = [deltaE76(candidate, r) for r in refs]
    return {"cr_mean":sum(crs)/len(crs), "cr_min":min(crs), "de_mean":sum(des)/len(des), "de_min":min(des)}

# --- main recommend function ---
def recommend(
    primaries: List[Any],             # list of rgb/hex or (rgb/hex, proportion)
    secondaries: Optional[List[Any]] = None,
    cvd_profile: Optional[Dict[str,Any]] = None,
    topk: int = 5,
    context: Optional[Dict[str,Any]] = None,
    ml_model: Optional[Any] = None,   # optional sklearn/lightgbm model for re-ranking (must accept feature vectors)
    feedback_log: Optional[str] = "reco_feedback.csv"
) -> List[Dict[str,Any]]:
    """
    Returns top-K recommended accent colors (hex + metrics + explanation).
    Accepts custom cvd_profile (transform_matrix/preference). If ml_model given, uses it to re-rank.
    """
    context = context or {}
    prims = expand_color_inputs(primaries)
    secs  = expand_color_inputs(secondaries or [])
    # flatten to color lists for operations, with weights
    prim_colors = [p for p,_ in prims]
    prim_weights = [w for _,w in prims]
    sec_colors = [s for s,_ in secs]
    sec_weights = [w for _,w in secs]

    # build cvd transform
    cvd_mat = build_cvd_matrix_from_profile(cvd_profile or {"type":"normal","severity":0.0})

    # candidate pool: theory-derived from primaries and cross-blends
    pool = []
    for c in prim_colors:
        pool += generate_theory_candidates(c)
    # cross-blend midpoints
    if len(prim_colors) >= 2:
        for a,b in itertools.combinations(prim_colors,2):
            ha,sa,va = rgb_to_hsv_deg(a); hb,sb,vb = rgb_to_hsv_deg(b)
            hmid = (ha+hb)/2.0; smid = (sa+sb)/2.0; vmid = (va+vb)/2.0
            pool += generate_theory_candidates(hsv_deg_to_rgb(hmid, smid, vmid))
    # dedupe by ΔE
    uniq = []
    for c in pool:
        if all(deltaE76(c,u) > 5.0 for u in uniq):
            uniq.append(c)
    pool = uniq

    # filter: don't propose colors too close to existing palette
    filtered = [c for c in pool if all(deltaE76(c, e) > 8.0 for e in prim_colors+sec_colors)]
    if not filtered:
        filtered = pool  # fallback if too strict

    # score each candidate
    scored = []
    for cand in filtered:
        # normal harmony
        harmony_norm = harmony_score_for_candidate(cand, prim_colors)
        # simulate candidate and refs under cvd matrix
        cand_sim = apply_matrix_to_rgb(cand, cvd_mat)
        prims_sim = [apply_matrix_to_rgb(p, cvd_mat) for p in prim_colors]
        secs_sim  = [apply_matrix_to_rgb(s, cvd_mat) for s in sec_colors]
        # contrast & deltaE under simulated (distinguishability)
        cb_prim = aggregate_contrast_metrics(cand_sim, prims_sim)
        cb_sec  = aggregate_contrast_metrics(cand_sim, secs_sim) if secs_sim else {"cr_mean":0,"de_mean":0,"cr_min":0,"de_min":0}
        # also normal contrast as regularizer
        cb_norm = aggregate_contrast_metrics(cand, prim_colors + sec_colors)
        # build feature vector for optional ML too (flat)
        feature_vector = [
            harmony_norm,
            cb_prim["de_mean"], cb_prim["de_min"], cb_prim["cr_mean"], cb_prim["cr_min"],
            cb_sec["de_mean"], cb_sec["de_min"], cb_sec["cr_mean"], cb_sec["cr_min"],
            cb_norm["de_mean"], cb_norm["cr_mean"]
        ]
        # scoring: weights (tunable)
        score = (
            0.45 * harmony_norm +
            0.40 * (0.6 * (cb_prim["de_mean"]/50.0) + 0.4 * (cb_prim["cr_mean"]/7.0)) +
            0.10 * (0.6 * (cb_sec["de_mean"]/50.0) + 0.4*(cb_sec["cr_mean"]/7.0)) +
            0.05 * (0.5*(cb_norm["de_mean"]/50.0) + 0.5*(cb_norm["cr_mean"]/7.0))
        )
        scored.append({
            "rgb": cand,
            "hex": rgb_to_hex(cand),
            "score": score,
            "harmony_norm": harmony_norm,
            "prim_sim_metrics": cb_prim,
            "sec_sim_metrics": cb_sec,
            "norm_metrics": cb_norm,
            "features": feature_vector
        })

    # optional ML rerank if model provided
    if ml_model is not None:
        # ml_model.predict_proba or predict; expect shape (n_samples, n_features)
        X = [s["features"] for s in scored]
        try:
            if hasattr(ml_model, "predict_proba"):
                probs = ml_model.predict_proba(X)[:,1]
                for s,p in zip(scored, probs):
                    s["ml_score"] = float(p)
                scored.sort(key=lambda d: d["ml_score"], reverse=True)
            else:
                preds = ml_model.predict(X)
                for s,p in zip(scored, preds):
                    s["ml_score"] = float(p)
                scored.sort(key=lambda d: d["ml_score"], reverse=True)
        except Exception as e:
            # model failed; fallback to rule scoring
            print("ML re-rank failed:", e)
            scored.sort(key=lambda d: d["score"], reverse=True)
    else:
        scored.sort(key=lambda d: d["score"], reverse=True)

    top = scored[:topk]
    # add textual explanation & CVD safety flag based on thresholds in cvd_profile
    min_de = cvd_profile.get("min_cvd_deltaE", 10.0) if cvd_profile else 10.0
    min_cr = cvd_profile.get("min_cvd_contrast", 1.2) if cvd_profile else 1.2
    for s in top:
        prim_de = s["prim_sim_metrics"]["de_min"]
        prim_cr = s["prim_sim_metrics"]["cr_min"]
        ok_de = prim_de >= min_de
        ok_cr = prim_cr >= min_cr
        s["cvd_safe"] = bool(ok_de and ok_cr)
        s["explanation"] = (
            f"Harmony score (normal): {s['harmony_norm']:.2f}; "
            f"Sim ΔE mean (prim): {s['prim_sim_metrics']['de_mean']:.1f}, min: {prim_de:.1f}; "
            f"Sim CR mean: {s['prim_sim_metrics']['cr_mean']:.2f}. "
            + ("OK for CVD" if s["cvd_safe"] else "May be indistinguishable under CVD")
        )

    # Log candidate choices for audit / dataset building if feedback_log provided
    if feedback_log:
        header = ["primaries","secondaries","cvd_profile","context","candidate_hex","score","cvd_safe"]
        file_exists = os.path.exists(feedback_log)
        try:
            with open(feedback_log,"a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(header)
                for s in top:
                    writer.writerow([
                        json.dumps([rgb_to_hex(c) for c in prim_colors]),
                        json.dumps([rgb_to_hex(c) for c in sec_colors]),
                        json.dumps(cvd_profile or {}),
                        json.dumps(context or {}),
                        s["hex"], f"{s['score']:.4f}", int(s["cvd_safe"])
                    ])
        except Exception as e:
            print("Feedback log failed:", e)

    return top

# --- Demo usage if run as main ---
if __name__ == "__main__":
    # Example inputs
    prim = [("#1F3A93", 0.7), ("#FFFFFF",0.3)]   # navy + white (proportions)
    sec = ["#D2B48C"]                            # tan
    # Suppose the CVD detection pipeline returns a custom profile:
    cvd_profile = {
        "type":"deuteranopia",
        "severity":0.85,
        # optional: provide exact transform_matrix computed by the detection pipeline:
        # "transform_matrix": [[...],[...],[...]],
        "min_cvd_deltaE": 12.0,
        "min_cvd_contrast": 1.25,
        "cone_sensitivities": {"L":0.9,"M":0.6,"S":1.0}
    }
    recs = recommend(prim, sec, cvd_profile=cvd_profile, topk=5, context={"occasion":"casual"})
    print("Top recommendations:")
    for r in recs:
        print(r["hex"], r["cvd_safe"], r["explanation"])
