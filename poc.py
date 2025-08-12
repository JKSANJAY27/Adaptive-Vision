import sys
import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from matplotlib import pyplot as plt
from matplotlib import colors as mpl_colors
import seaborn as sns

# -------------------------
# Utility: GrabCut segmentation
# -------------------------
def grabcut_segment(image_path, rect_margin=10, iter_count=5):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (rect_margin, rect_margin, w - 2*rect_margin, h - 2*rect_margin)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_rgb, mask, rect, bgModel, fgModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = img_rgb * mask2[:, :, np.newaxis]
    return img_rgb, segmented, mask2

# -------------------------
# Utility: Dominant color via KMeans
# -------------------------
def extract_dominant_color(segmented_rgb, clusters=1):
    # flatten pixels and remove fully black background from segmentation
    pixels = segmented_rgb.reshape(-1, 3)
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
    if len(pixels) == 0:
        # fallback to whole image if segmentation failed
        pixels = segmented_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(pixels)
    dominant = kmeans.cluster_centers_[0].astype(int)
    return dominant

# -------------------------
# Fuzzy logic: describe hue/sat/value
# -------------------------
def build_fuzzy_system():
    # Inputs: hue 0..360, sat 0..100, val 0..100
    hue_var = ctrl.Antecedent(np.arange(0, 361, 1), 'hue')
    sat_var = ctrl.Antecedent(np.arange(0, 101, 1), 'saturation')
    val_var = ctrl.Antecedent(np.arange(0, 101, 1), 'value')

    # Output: descriptive score (0..100) — we will also read membership degrees directly
    score_var = ctrl.Consequent(np.arange(0, 101, 1), 'score')

    # hue: warm / neutral / cool using Gaussians
    hue_var['warm'] = fuzz.gaussmf(hue_var.universe, 0, 40) + fuzz.gaussmf(hue_var.universe, 360, 40)
    hue_var['neutral'] = fuzz.gaussmf(hue_var.universe, 120, 40)
    hue_var['cool'] = fuzz.gaussmf(hue_var.universe, 240, 40)

    # saturation: muted / vivid
    sat_var['muted'] = fuzz.gaussmf(sat_var.universe, 25, 18)
    sat_var['vivid'] = fuzz.gaussmf(sat_var.universe, 75, 18)

    # value: dark / bright
    val_var['dark'] = fuzz.gaussmf(val_var.universe, 20, 18)
    val_var['bright'] = fuzz.gaussmf(val_var.universe, 80, 18)

    # score for demonstration: low/high
    score_var['low'] = fuzz.trimf(score_var.universe, [0, 0, 50])
    score_var['high'] = fuzz.trimf(score_var.universe, [50, 100, 100])

    # Rules (simple interpretable ones)
    r1 = ctrl.Rule(hue_var['warm'] & sat_var['vivid'] & val_var['bright'], score_var['high'])
    r2 = ctrl.Rule(hue_var['cool'] & sat_var['muted'] & val_var['dark'], score_var['low'])
    r3 = ctrl.Rule(hue_var['neutral'], score_var['high'])
    system = ctrl.ControlSystem([r1, r2, r3])
    sim = ctrl.ControlSystemSimulation(system)
    return (hue_var, sat_var, val_var, score_var, sim)

def fuzzy_describe(hue, sat, val, fuzzy_components):
    hue_var, sat_var, val_var, score_var, sim = fuzzy_components
    # compute membership degrees manually for richer description
    hue_mems = {label: fuzz.interp_membership(hue_var.universe, hue_var[label].mf, hue) for label in hue_var.terms}
    sat_mems = {label: fuzz.interp_membership(sat_var.universe, sat_var[label].mf, sat) for label in sat_var.terms}
    val_mems = {label: fuzz.interp_membership(val_var.universe, val_var[label].mf, val) for label in val_var.terms}

    # run the control simulation to get single numeric score
    sim.input['hue'] = hue
    sim.input['saturation'] = sat
    sim.input['value'] = val
    sim.compute()
    score = sim.output['score']

    # convert memberships to textual labels taking highest membership
    hue_label = max(hue_mems.items(), key=lambda x: x[1])[0]
    sat_label = max(sat_mems.items(), key=lambda x: x[1])[0]
    val_label = max(val_mems.items(), key=lambda x: x[1])[0]

    desc_text = f"{hue_label.capitalize()}, {sat_label.capitalize()}, {val_label.capitalize()}"
    return desc_text, score, {'hue_mems': hue_mems, 'sat_mems': sat_mems, 'val_mems': val_mems}

# -------------------------
# Recommendation generation (complementary)
# -------------------------
def complementary_color_hex_from_hsv(h, s, v):
    # h in degrees [0,360], s,v in [0,100]
    comp_h = (h + 180) % 360
    hsv = (comp_h/360.0, s/100.0, v/100.0)
    rgb = mpl_colors.hsv_to_rgb([[hsv]])[0][0]
    rgb255 = (rgb * 255).astype(int)
    return rgb255

# -------------------------
# Synthetic dataset generation for color harmony
# -------------------------
def harmony_label_by_heuristic(h1, s1, v1, h2, s2, v2):
    """
    Heuristic labelling:
    Good match if:
      - hue diff approx 0..30 (analogous) OR approx 180 +/- 30 (complementary) OR approx 120 +/-25 (triadic)
      - AND saturation difference small (<30)
      - AND value (brightness) difference small (<30)
    Otherwise BAD.
    """
    # hue diff in circular space
    dh = abs((h1 - h2 + 180) % 360 - 180)
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)
    hue_good = (dh <= 30) or (abs(dh - 180) <= 30) or (abs(dh - 120) <= 25)
    sat_good = ds <= 30
    val_good = dv <= 30
    score = 1 if (hue_good and sat_good and val_good) else 0
    return score

def generate_color_pairs_dataset(n_pairs=5000, random_state=0):
    rng = np.random.RandomState(random_state)
    X = []
    y = []
    for _ in range(n_pairs):
        # sample color 1 and color 2 in HSV space
        h1 = rng.uniform(0, 360)
        s1 = rng.uniform(10, 100)  # avoid super low saturation fully gray
        v1 = rng.uniform(10, 100)
        # produce color 2 with some bias toward similarity or complementary randomly
        if rng.rand() < 0.5:
            # similar-ish
            h2 = (h1 + rng.normal(0, 20)) % 360
            s2 = np.clip(s1 + rng.normal(0, 15), 0, 100)
            v2 = np.clip(v1 + rng.normal(0, 15), 0, 100)
        else:
            # random / diverse
            h2 = rng.uniform(0, 360)
            s2 = rng.uniform(10, 100)
            v2 = rng.uniform(10, 100)
        dh = abs((h1 - h2 + 180) % 360 - 180)
        ds = abs(s1 - s2)
        dv = abs(v1 - v2)
        label = harmony_label_by_heuristic(h1, s1, v1, h2, s2, v2)
        X.append([dh, ds, dv])
        y.append(label)
    return np.array(X), np.array(y)

# -------------------------
# Train ML model (RandomForest) and evaluate
# -------------------------
def train_and_evaluate_model(X, y, test_size=0.2, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return clf, acc, cm, (X_test, y_test, y_pred)

# -------------------------
# Visualization helpers
# -------------------------
def show_color_patch(rgb, title="color"):
    patch = np.ones((50, 150, 3), dtype=np.uint8) * rgb.reshape(1,1,3).astype(np.uint8)
    plt.imshow(patch)
    plt.axis('off')
    plt.title(title)

def visualize_segmentation(original, segmented, dominant_rgb, recommended_rgb, pred_label, score, desc_text):
    plt.figure(figsize=(12,6))
    plt.subplot(2,3,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis('off')

    plt.subplot(2,3,2)
    plt.imshow(segmented)
    plt.title("Segmented (GrabCut)")
    plt.axis('off')

    plt.subplot(2,3,3)
    show_color_patch(dominant_rgb, title=f"Dominant: {tuple(dominant_rgb)}")
    plt.subplot(2,3,4)
    show_color_patch(recommended_rgb, title=f"Recommended: {tuple(recommended_rgb)}")

    plt.subplot(2,3,5)
    plt.text(0.01, 0.5, f"Fuzzy Description: {desc_text}\nFuzzy Score: {score:.2f}\nML Prediction: {'GOOD' if pred_label==1 else 'BAD'}", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# -------------------------
# Main pipeline
# -------------------------
def run_pipeline(image_path):
    print("Starting pipeline...")
    original, segmented, mask = grabcut_segment(image_path)
    dominant_rgb = extract_dominant_color(segmented)
    # convert dominant RGB to HSV (matplotlib uses 0..1)
    dominant_hsv = mpl_colors.rgb_to_hsv([[dominant_rgb/255.0]])[0][0]
    hue = dominant_hsv[0]*360.0
    sat = dominant_hsv[1]*100.0
    val = dominant_hsv[2]*100.0

    # fuzzy description
    fuzzy_components = build_fuzzy_system()
    desc_text, fuzzy_score, mems = fuzzy_describe(hue, sat, val, fuzzy_components)
    print(f"Dominant RGB: {dominant_rgb}, HSV (h,s,v): ({hue:.1f}, {sat:.1f}, {val:.1f})")
    print("Fuzzy description:", desc_text, f" (score={fuzzy_score:.2f})")

    # recommended complementary color
    rec_rgb = complementary_color_hex_from_hsv(hue, sat, val)

    # train ML model on synthetic dataset
    print("Generating synthetic dataset and training ML model...")
    X, y = generate_color_pairs_dataset(5000, random_state=42)
    clf, acc, cm, test_info = train_and_evaluate_model(X, y, test_size=0.2, random_state=42)
    print(f"Model accuracy on synthetic test set: {acc*100:.2f}%")
    print("Confusion matrix:\n", cm)

    # now compute features comparing dominant vs recommended
    # convert rec_rgb to HSV
    rec_hsv = mpl_colors.rgb_to_hsv([[rec_rgb/255.0]])[0][0]
    rec_h, rec_s, rec_v = rec_hsv[0]*360.0, rec_hsv[1]*100.0, rec_hsv[2]*100.0
    dh = abs((hue - rec_h + 180) % 360 - 180)
    ds = abs(sat - rec_s)
    dv = abs(val - rec_v)
    pred = clf.predict([[dh, ds, dv]])[0]

    # visualize
    visualize_segmentation(original, segmented, dominant_rgb, rec_rgb, pred, fuzzy_score, desc_text)

    return {
        'dominant_rgb': dominant_rgb.tolist(),
        'dominant_hsv': (hue, sat, val),
        'fuzzy_desc': desc_text,
        'fuzzy_score': fuzzy_score,
        'recommended_rgb': rec_rgb.tolist(),
        'ml_prediction': 'GOOD' if pred==1 else 'BAD',
        'model_accuracy': acc,
        'confusion_matrix': cm.tolist()
    }

# -------------------------
# Run script
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python color_poc.py /path/to/image.jpg")
        sys.exit(1)
    image_path = 'shirt.jpg'
    if not os.path.exists(image_path):
        print("Image not found:", image_path)
        sys.exit(1)
    results = run_pipeline(image_path)
    print("\n--- Summary ---")
    for k, v in results.items():
        print(f"{k}: {v}")
