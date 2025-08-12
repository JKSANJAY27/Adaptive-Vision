import os
import io
import json
import math
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for, send_file
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import colors as mpl_colors
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
import base64
from flask import session
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
PROFILE_PATH = 'user_profile.json'
# simple dev secret; replace with a persistent secure value in production
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-replace-me-please')


# -------------------------
# Optional Mask-RCNN loader (OpenCV DNN). Will fallback to GrabCut if files missing.
# -------------------------
MASK_PB = 'frozen_inference_graph.pb'
MASK_PBTXT = 'mask_rcnn_inception_v2_coco_2018_01_28_opencv.pbtxt'
COCO_LABELS = 'coco_labels.txt'

mask_net = None
coco_names = None
if os.path.exists(MASK_PB) and os.path.exists(MASK_PBTXT) and os.path.exists(COCO_LABELS):
    try:
        mask_net = cv2.dnn.readNetFromTensorflow(MASK_PB, MASK_PBTXT)
        with open(COCO_LABELS, 'r') as f:
            coco_names = [l.strip() for l in f.readlines()]
        print("Mask R-CNN loaded successfully.")
    except Exception as e:
        print("Mask R-CNN load error:", e)
        mask_net = None
else:
    print("Mask R-CNN files not found — falling back to GrabCut segmentation.")

# -------------------------
# Utility: Save & load simple profile (hue_offset, sat_scale)
# -------------------------
def save_profile(profile):
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile, f)

def load_profile():
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, 'r') as f:
            return json.load(f)
    return {'hue_offset': 0.0, 'sat_scale': 1.0}

# -------------------------
# Mask-RCNN person segmentation (returns mask with 1 for person)
# -------------------------
def maskrcnn_person_mask(image_rgb):
    if mask_net is None:
        return None
    h, w = image_rgb.shape[:2]
    blob = cv2.dnn.blobFromImage(image_rgb, swapRB=True, crop=False)
    mask_net.setInput(blob)
    boxes, masks = mask_net.forward(['detection_out_final', 'detection_masks'])
    # boxes: [1,1,N,7] ; masks: [1,N,80,15,15]
    boxes = boxes[0,0]
    masks = masks[0]
    person_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(boxes.shape[0]):
        score = float(boxes[i,2])
        class_id = int(boxes[i,1])
        if score < 0.5:
            continue
        # check if class is person (COCO person id typically 1)
        label = coco_names[class_id] if coco_names and class_id < len(coco_names) else str(class_id)
        if label != 'person':
            continue
        x1 = int(boxes[i,3] * w)
        y1 = int(boxes[i,4] * h)
        x2 = int(boxes[i,5] * w)
        y2 = int(boxes[i,6] * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        mask = masks[i, class_id]
        # resize mask to bounding box size
        mask = cv2.resize(mask, (x2 - x1 + 1, y2 - y1 + 1))
        mask = (mask > 0.5).astype(np.uint8)
        person_mask[y1:y2+1, x1:x2+1] = np.maximum(person_mask[y1:y2+1, x1:x2+1], mask)
    if person_mask.sum() == 0:
        return None
    return person_mask

# -------------------------
# GrabCut fallback segmentation
# -------------------------
def grabcut_segment(image_rgb, rect_margin=10, iter_count=5):
    h, w = image_rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (rect_margin, rect_margin, w - 2*rect_margin, h - 2*rect_margin)
    bgModel = np.zeros((1,65), np.float64)
    fgModel = np.zeros((1,65), np.float64)
    try:
        cv2.grabCut(image_rgb, mask, rect, bgModel, fgModel, iter_count, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
        return mask2
    except Exception as e:
        print("GrabCut failed:", e)
        return None

# -------------------------
# Dominant color extraction (KMeans)
# -------------------------
def extract_dominant_color_from_mask(image_rgb, mask, clusters=1):
    pixels = image_rgb.reshape(-1,3)
    if mask is not None:
        flat_mask = mask.reshape(-1)
        pixels = pixels[flat_mask==1]
    # fallback
    if len(pixels)==0:
        pixels = image_rgb.reshape(-1,3)
    # For speed downsample if too large
    if len(pixels) > 15000:
        idx = np.random.choice(len(pixels), 15000, replace=False)
        pixels = pixels[idx]
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(pixels)
    return kmeans.cluster_centers_[0].astype(int)

# -------------------------
# Fuzzy system builder (same structure as earlier)
# -------------------------
def build_fuzzy_system():
    hue_var = ctrl.Antecedent(np.arange(0,361,1), 'hue')
    sat_var = ctrl.Antecedent(np.arange(0,101,1), 'saturation')
    val_var = ctrl.Antecedent(np.arange(0,101,1), 'value')
    score_var = ctrl.Consequent(np.arange(0,101,1), 'score')

    hue_var['warm'] = fuzz.gaussmf(hue_var.universe, 0, 40) + fuzz.gaussmf(hue_var.universe, 360, 40)
    hue_var['neutral'] = fuzz.gaussmf(hue_var.universe, 120, 40)
    hue_var['cool'] = fuzz.gaussmf(hue_var.universe, 240, 40)

    sat_var['muted'] = fuzz.gaussmf(sat_var.universe, 25, 18)
    sat_var['vivid'] = fuzz.gaussmf(sat_var.universe, 75, 18)
    val_var['dark'] = fuzz.gaussmf(val_var.universe, 20, 18)
    val_var['bright'] = fuzz.gaussmf(val_var.universe, 80, 18)
    score_var['low'] = fuzz.trimf(score_var.universe, [0,0,50])
    score_var['high'] = fuzz.trimf(score_var.universe, [50,100,100])

    r1 = ctrl.Rule(hue_var['warm'] & sat_var['vivid'] & val_var['bright'], score_var['high'])
    r2 = ctrl.Rule(hue_var['cool'] & sat_var['muted'] & val_var['dark'], score_var['low'])
    r3 = ctrl.Rule(hue_var['neutral'], score_var['high'])
    system = ctrl.ControlSystem([r1,r2,r3])
    sim = ctrl.ControlSystemSimulation(system)
    return (hue_var, sat_var, val_var, score_var, sim)

def fuzzy_describe(hue, sat, val, fuzzy_components):
    hue_var, sat_var, val_var, score_var, sim = fuzzy_components
    hue_mems = {label: fuzz.interp_membership(hue_var.universe, hue_var[label].mf, hue) for label in hue_var.terms}
    sat_mems = {label: fuzz.interp_membership(sat_var.universe, sat_var[label].mf, sat) for label in sat_var.terms}
    val_mems = {label: fuzz.interp_membership(val_var.universe, val_var[label].mf, val) for label in val_var.terms}

    sim.input['hue'] = hue
    sim.input['saturation'] = sat
    sim.input['value'] = val
    sim.compute()
    score = sim.output['score']
    hue_label = max(hue_mems.items(), key=lambda x: x[1])[0]
    sat_label = max(sat_mems.items(), key=lambda x: x[1])[0]
    val_label = max(val_mems.items(), key=lambda x: x[1])[0]
    desc_text = f"{hue_label.capitalize()}, {sat_label.capitalize()}, {val_label.capitalize()}"
    return desc_text, score, {'hue_mems': hue_mems, 'sat_mems': sat_mems, 'val_mems': val_mems}

# -------------------------
# Complementary color
# -------------------------
def complementary_rgb_from_hsv(h, s, v):
    ch = (h + 180) % 360
    rgb = mpl_colors.hsv_to_rgb([[(ch/360.0, s/100.0, v/100.0)]])[0][0]
    return (rgb*255).astype(int)

# -------------------------
# Synthetic dataset and ML training (RandomForest)
# -------------------------
def harmony_label_by_heuristic(h1,s1,v1,h2,s2,v2):
    dh = abs((h1-h2+180)%360 - 180)
    ds = abs(s1-s2)
    dv = abs(v1-v2)
    hue_good = (dh <= 30) or (abs(dh-180) <= 30) or (abs(dh-120) <= 25)
    sat_good = ds <= 30
    val_good = dv <= 30
    return 1 if (hue_good and sat_good and val_good) else 0

def generate_pairs(n_pairs=3000, seed=0):
    rng = np.random.RandomState(seed)
    X = []
    y = []
    for _ in range(n_pairs):
        h1 = rng.uniform(0,360); s1 = rng.uniform(10,100); v1 = rng.uniform(10,100)
        if rng.rand() < 0.5:
            h2 = (h1 + rng.normal(0,20)) % 360
            s2 = np.clip(s1 + rng.normal(0,15), 0, 100)
            v2 = np.clip(v1 + rng.normal(0,15), 0, 100)
        else:
            h2 = rng.uniform(0,360); s2 = rng.uniform(10,100); v2 = rng.uniform(10,100)
        dh = abs((h1-h2+180)%360 - 180); ds = abs(s1-s2); dv = abs(v1-v2)
        X.append([dh, ds, dv]); y.append(harmony_label_by_heuristic(h1,s1,v1,h2,s2,v2))
    return np.array(X), np.array(y)

def train_model():
    X, y = generate_pairs(4000, seed=42)
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    cm = confusion_matrix(yte, ypred)
    return clf, acc, cm

# Train once at startup
MODEL, MODEL_ACC, MODEL_CM = train_model()

# -------------------------
# Helper: image -> base64 for inline display
# -------------------------
def pil_to_base64_img(img_pil):
    buff = io.BytesIO()
    img_pil.save(buff, format='PNG')
    b64 = base64.b64encode(buff.getvalue()).decode('utf-8')
    return 'data:image/png;base64,' + b64

# -------------------------
# Main pipeline that integrates everything
# -------------------------
def process_image_and_recommend(img_path, profile):
    # load image (RGB)
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Attempt mask-rcnn person detection
    person_mask = maskrcnn_person_mask(img_rgb) if mask_net is not None else None

    # If person mask exists, use it, else fallback to grabcut
    if person_mask is None:
        print("Using GrabCut fallback.")
        person_mask = grabcut_segment(img_rgb)
    else:
        print("Using Mask R-CNN person mask.")

    # refine mask: keep largest connected component to reduce noise
    if person_mask is not None:
        person_mask = person_mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(person_mask, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # ignore background label 0
            refined = (labels == largest).astype(np.uint8)
            person_mask = refined

    dominant_rgb = extract_dominant_color_from_mask(img_rgb, person_mask)
    # convert to HSV
    dominant_hsv = mpl_colors.rgb_to_hsv([[dominant_rgb/255.0]])[0][0]
    hue = dominant_hsv[0]*360.0; sat = dominant_hsv[1]*100.0; val = dominant_hsv[2]*100.0

    # apply user calibration adjustments
    hue = (hue + profile.get('hue_offset', 0.0)) % 360.0
    sat = np.clip(sat * profile.get('sat_scale', 1.0), 0, 100)

    fuzzy_components = build_fuzzy_system()
    desc_text, fuzzy_score, mems = fuzzy_describe(hue, sat, val, fuzzy_components)

    rec_rgb = complementary_rgb_from_hsv(hue, sat, val)
    rec_hsv = mpl_colors.rgb_to_hsv([[rec_rgb/255.0]])[0][0]
    rec_h = rec_hsv[0]*360.0; rec_s = rec_hsv[1]*100.0; rec_v = rec_hsv[2]*100.0
    dh = abs((hue - rec_h + 180)%360 - 180); ds = abs(sat - rec_s); dv = abs(val - rec_v)
    pred = MODEL.predict([[dh, ds, dv]])[0]

    # prepare images for inline display: segmented area overlay
    overlay = img_rgb.copy()
    if person_mask is not None:
        overlay_mask = (person_mask==1)
        overlay[~overlay_mask] = (overlay[~overlay_mask] * 0.25).astype(np.uint8)
    pil_overlay = Image.fromarray(overlay)
    overlay_b64 = pil_to_base64_img(pil_overlay)

    dominant_patch = Image.new('RGB', (150,100), tuple(map(int, dominant_rgb)))
    rec_patch = Image.new('RGB', (150,100), tuple(map(int, rec_rgb)))
    dom_b64 = pil_to_base64_img(dominant_patch)
    rec_b64 = pil_to_base64_img(rec_patch)

    return {
        'overlay_b64': overlay_b64,
        'dominant_rgb': dominant_rgb.tolist(),
        'dominant_hsv': (round(hue,1), round(sat,1), round(val,1)),
        'dominant_patch': dom_b64,
        'rec_rgb': rec_rgb.tolist(),
        'rec_patch': rec_b64,
        'fuzzy_desc': desc_text,
        'fuzzy_score': round(float(fuzzy_score),2),
        'ml_pred': 'GOOD' if pred==1 else 'BAD',
        'model_acc': round(float(MODEL_ACC),3),
        'model_cm': MODEL_CM.tolist()
    }

# -------------------------
# Routes
# -------------------------
@app.route('/')
def index():
    profile = load_profile()
    return render_template('index.html', profile=profile)

CALIBRATION_TRIALS = 5  # change if you want more/less trials

@app.route('/calibrate', methods=['GET', 'POST'])
def calibrate():
    # initialize session container (if not present)
    if 'calib' not in session:
        session['calib'] = {'index': 0, 'data': []}

    # POST: user selected one option for current trial
    if request.method == 'POST':
        try:
            ref_h = float(request.form.get('ref_h', 0.0))
            ref_s = float(request.form.get('ref_s', 50.0))
            sel_h = float(request.form.get('sel_h', ref_h))
            sel_s = float(request.form.get('sel_s', ref_s))

            hue_offset = (sel_h - ref_h + 180) % 360 - 180
            sat_scale = (sel_s / ref_s) if (ref_s > 0) else 1.0

            # safe modify session (read -> mutate -> reassign)
            tmp = session['calib']
            tmp['data'].append({'hue_offset': float(hue_offset), 'sat_scale': float(sat_scale)})
            tmp['index'] = int(tmp.get('index', 0)) + 1
            session['calib'] = tmp
        except Exception as e:
            print("Calibration POST error:", e)
            return redirect(url_for('index'))

    # If finished all trials, compute averages, save profile and clear session data
    if session['calib']['index'] >= CALIBRATION_TRIALS:
        data = session['calib']['data']
        offsets = [d['hue_offset'] for d in data]
        scales = [d['sat_scale'] for d in data]
        avg_offset = sum(offsets) / len(offsets)
        avg_scale = sum(scales) / len(scales)

        profile = {'hue_offset': round(avg_offset, 2), 'sat_scale': round(avg_scale, 3)}
        save_profile(profile)

        # clear calibration session state
        session.pop('calib', None)
        return render_template('calibrate_result.html', profile=profile)

    # Otherwise generate the next trial (GET or after POST)
    # create reference HSV + 3 decoys, compute RGBs for template
    ref_h = random.uniform(0, 360)
    ref_s = random.uniform(20, 90)
    ref_v = random.uniform(30, 90)

    # reference rgb (for display)
    ref_rgb = (mpl_colors.hsv_to_rgb([[(ref_h / 360.0, ref_s / 100.0, ref_v / 100.0)]])[0][0] * 255).astype(int)

    options = []
    # make 4 options; include the exact reference as one of them
    for i in range(4):
        if i == 0:
            h, s, v = ref_h, ref_s, ref_v
        else:
            h = (ref_h + random.uniform(-30, 30)) % 360
            s = float(np.clip(ref_s * random.uniform(0.6, 1.4), 0, 100))
            v = float(np.clip(ref_v * random.uniform(0.8, 1.2), 0, 100))
        rgb = (mpl_colors.hsv_to_rgb([[(h / 360.0, s / 100.0, v / 100.0)]])[0][0] * 255).astype(int)
        options.append({'h': float(h), 's': float(s), 'v': float(v), 'rgb': tuple(rgb)})

    # shuffle options so the correct choice isn't always the first button
    random.shuffle(options)

    trial_number = session['calib']['index'] + 1
    return render_template(
        'calibrate.html',
        trial_number=trial_number,
        total_trials=CALIBRATION_TRIALS,
        ref_h=ref_h,
        ref_s=ref_s,
        ref_rgb=tuple(ref_rgb),
        options=options
    )

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    f = request.files['image']
    fname = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(fname)
    profile = load_profile()
    results = process_image_and_recommend(fname, profile)
    return render_template('result.html', results=results)

# -------------------------
# Run app
# -------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
