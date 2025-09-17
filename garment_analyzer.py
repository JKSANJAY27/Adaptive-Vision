import os
import sys
import subprocess
import importlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from skimage.morphology import remove_small_objects
import torch

# Install required packages
def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

required = [
    "git+https://github.com/facebookresearch/segment-anything.git",
    "colormath",
    "opencv-python-headless",
    "scikit-image",
    "scikit-learn",
    "torch",
    "torchvision",
    "gdown"
]

for req in required:
    short = req.split("==")[0].split("+")[-1].split("/")[-1]
    try:
        importlib.import_module(short)
    except Exception:
        print(f"Installing: {req}")
        pip_install(req)

# Patch numpy.asscalar for compatibility
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item()

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

def download_sam_checkpoint():
    """Download SAM checkpoint if missing"""
    CKPT_NAME = "sam_vit_b.pth"
    CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    if not os.path.exists(CKPT_NAME):
        print("Downloading SAM checkpoint (vit_b)...")
        import urllib.request
        urllib.request.urlretrieve(CKPT_URL, CKPT_NAME)
        print("SAM checkpoint downloaded.")
    
    return CKPT_NAME

def load_sam_model():
    """Load SAM model robustly"""
    ckpt_path = download_sam_checkpoint()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM on {DEVICE}...")
    
    sam = sam_model_registry["vit_b"](checkpoint=None)
    try:
        state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        sam.load_state_dict(state)
    except TypeError:
        state = torch.load(ckpt_path, map_location=DEVICE)
        sam.load_state_dict(state)
    
    sam.to(DEVICE)
    sam.eval()
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("SAM ready.")
    
    return mask_generator

def segment_garment_sam(image_path):
    """Advanced garment segmentation using SAM"""
    # Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"Could not load image at {image_path}")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    
    # Load SAM
    try:
        mask_generator = load_sam_model()
        
        # Generate masks
        masks = mask_generator.generate(rgb)
        if not masks:
            print("SAM returned no masks, falling back to GrabCut...")
            return segment_garment_fallback(rgb)
        
        # Score masks for garment selection
        masks_sorted = sorted(masks, key=lambda m: m.get("area", 0), reverse=True)
        top_k = min(8, len(masks_sorted))
        candidates = masks_sorted[:top_k]
        
        # Compute LAB for chroma analysis
        img_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        cent_x, cent_y = W/2.0, H/2.0
        
        mask_infos = []
        max_area = 0
        max_chroma = 0.0
        
        for m in candidates:
            seg = m['segmentation'].astype(bool)
            area = int(seg.sum())
            if area == 0:
                continue
                
            max_area = max(max_area, area)
            
            # Mean chroma calculation
            lab_pixels = img_lab[seg]
            a = lab_pixels[:,1].astype(float) - 128.0
            b = lab_pixels[:,2].astype(float) - 128.0
            mean_chroma = float(np.mean(np.sqrt(a**2 + b**2)))
            max_chroma = max(max_chroma, mean_chroma)
            
            # Center score
            ys, xs = np.where(seg)
            cx, cy = float(xs.mean()), float(ys.mean())
            dx = (cx - cent_x) / W
            dy = (cy - cent_y) / H
            dist = math.sqrt(dx*dx + dy*dy)
            center_score = 1.0 - min(dist, 1.0)
            
            mask_infos.append({
                'seg': seg,
                'area': area,
                'mean_chroma': mean_chroma,
                'center_score': center_score
            })
        
        if not mask_infos:
            print("No valid masks found, falling back to GrabCut...")
            return segment_garment_fallback(rgb)
        
        # Score and select best mask
        for info in mask_infos:
            norm_area = info['area'] / (max_area + 1e-9)
            norm_chroma = info['mean_chroma'] / (max_chroma + 1e-9)
            score = 0.45*norm_area + 0.45*norm_chroma + 0.10*info['center_score']
            info['score'] = score
        
        best = max(mask_infos, key=lambda x: x['score'])
        mask_bool = best['seg']
        
        print(f"SAM segmentation: area={best['area']}, chroma={best['mean_chroma']:.2f}")
        
        # Refine mask
        mask_uint8 = (mask_bool.astype("uint8") * 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_bool_refined = mask_uint8.astype(bool)
        mask_bool_refined = remove_small_objects(mask_bool_refined, min_size=int(0.001 * H * W))
        
        return rgb, mask_bool_refined
        
    except Exception as e:
        print(f"SAM failed ({e}), using fallback segmentation...")
        return segment_garment_fallback(rgb)

def segment_garment_fallback(rgb):
    """Fallback segmentation using GrabCut (from original code)"""
    H, W, _ = rgb.shape
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # GrabCut segmentation
    mask = np.zeros((H, W), np.uint8)
    margin = min(H, W) // 10
    rect = (margin, margin, W-2*margin, H-2*margin)
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask_grabcut = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # Background removal
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 230])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    non_white_mask = cv2.bitwise_not(white_mask)
    
    combined_mask = cv2.bitwise_and(mask_grabcut, non_white_mask // 255)
    
    if np.sum(combined_mask) < 0.01 * H * W:
        combined_mask = mask_grabcut
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    # Find largest contour
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        final_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(final_mask, [largest_contour], 255)
        return rgb, final_mask.astype(bool)
    else:
        center_mask = np.zeros((H, W), dtype=bool)
        center_h, center_w = H//3, W//3
        center_mask[center_h:2*center_h, center_w:2*center_w] = True
        return rgb, center_mask

def extract_colors_advanced(rgb, garment_mask):
    """Advanced color extraction combining both approaches"""
    # Get masked pixels
    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab_pixels_all = lab_img[garment_mask]
    
    if lab_pixels_all.size == 0:
        raise ValueError("No garment pixels found")
    
    # Brightness filtering (less aggressive)
    L_vals = lab_pixels_all[:,0]
    keep_mask = (L_vals > 5) & (L_vals < 250)
    lab_pixels = lab_pixels_all[keep_mask]
    
    if lab_pixels.shape[0] == 0:
        lab_pixels = lab_pixels_all
    
    print(f"Using {len(lab_pixels):,} pixels for color analysis")
    
    # Separate chromatic and achromatic colors
    a_vals = lab_pixels[:,1].astype(float) - 128.0
    b_vals = lab_pixels[:,2].astype(float) - 128.0
    pixel_chroma = np.sqrt(a_vals*a_vals + b_vals*b_vals)
    
    CHROMA_THRESHOLD = 8.0
    chromatic_mask = pixel_chroma >= CHROMA_THRESHOLD
    achromatic_mask = ~chromatic_mask
    
    lab_chromatic = lab_pixels[chromatic_mask]
    lab_achromatic = lab_pixels[achromatic_mask]
    
    print(f"Chromatic: {len(lab_chromatic)}, Achromatic: {len(lab_achromatic)}")
    
    # Process chromatic colors with clustering
    chromatic_centers = []
    if len(lab_chromatic) > 0:
        MAX_SAMPLE = 3000
        if len(lab_chromatic) > MAX_SAMPLE:
            rng = np.random.default_rng(42)
            samp_idx = rng.choice(len(lab_chromatic), size=MAX_SAMPLE, replace=False)
            lab_sample = lab_chromatic[samp_idx].astype(float)
        else:
            lab_sample = lab_chromatic.astype(float)
        
        # Try MeanShift first
        def try_meanshift(sample, quantiles=(0.15, 0.25, 0.10, 0.05)):
            for q in quantiles:
                try:
                    bw = estimate_bandwidth(sample, quantile=q, n_samples=min(500, len(sample)))
                    if not np.isfinite(bw) or bw <= 0:
                        continue
                    ms = MeanShift(bandwidth=bw, bin_seeding=True)
                    ms.fit(sample)
                    return ms.cluster_centers_
                except Exception:
                    continue
            return None
        
        centers = try_meanshift(lab_sample)
        if centers is None:
            # Fallback to KMeans
            K = min(4, max(2, len(lab_sample) // 1000))
            print(f"MeanShift failed, using KMeans(k={K})")
            km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(lab_sample)
            centers = km.cluster_centers_
        else:
            print(f"MeanShift found {len(centers)} chromatic centers")
        
        chromatic_centers = centers
    
    # Process achromatic colors
    achromatic_centers = []
    if len(lab_achromatic) > 0:
        L_achro = lab_achromatic[:,0]
        lightness_ranges = [
            (220, 255, "white"),
            (180, 219, "light_gray"),
            (120, 179, "medium_gray"),
            (60, 119, "dark_gray"),
            (0, 59, "black")
        ]
        
        for l_min, l_max, name in lightness_ranges:
            range_mask = (L_achro >= l_min) & (L_achro <= l_max)
            if range_mask.sum() > len(lab_achromatic) * 0.05:
                range_pixels = lab_achromatic[range_mask]
                median_center = np.median(range_pixels, axis=0)
                achromatic_centers.append(median_center)
                print(f"Found {name} color")
    
    # Combine all centers
    all_centers = []
    if len(chromatic_centers) > 0:
        all_centers.extend(chromatic_centers)
    if len(achromatic_centers) > 0:
        all_centers.extend(achromatic_centers)
    
    if len(all_centers) == 0:
        raise RuntimeError("No color centers found")
    
    all_centers = np.array(all_centers)
    
    # Assign pixels and compute proportions
    all_lab = lab_pixels.astype(float)
    closest_idx, _ = pairwise_distances_argmin_min(all_lab, all_centers)
    counts = Counter(closest_idx)
    total = sum(counts.values())
    
    # Sort by frequency
    center_items = sorted(counts.items(), key=lambda x: -x[1])
    center_indices = [ci for ci, _ in center_items]
    center_counts = [cnt for _, cnt in center_items]
    center_props = np.array(center_counts, dtype=float) / total
    
    return all_centers, center_props, center_indices

def lab_opencv_to_rgb_tuple(lab_opencv):
    """Convert OpenCV LAB to RGB tuple"""
    lab_arr = np.uint8([[lab_opencv]])
    rgb_val = cv2.cvtColor(lab_arr, cv2.COLOR_Lab2RGB)[0,0]
    return (int(rgb_val[0]), int(rgb_val[1]), int(rgb_val[2]))

def lab_opencv_to_labcolor(lab_opencv):
    """Convert OpenCV LAB to colormath LAB"""
    L = float(lab_opencv[0]) * 100.0 / 255.0
    a = float(lab_opencv[1]) - 128.0
    b = float(lab_opencv[2]) - 128.0
    return LabColor(L, a, b)

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex code"""
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

def get_color_name_simple(rgb):
    """Simple color naming"""
    r, g, b = rgb
    
    if r > 240 and g > 240 and b > 240:
        return "White"
    elif r < 40 and g < 40 and b < 40:
        return "Black"
    elif abs(r-g) < 20 and abs(g-b) < 20 and abs(r-b) < 20:
        if r > 200:
            return "Light Gray"
        elif r > 130:
            return "Gray"
        else:
            return "Dark Gray"
    elif r > g and r > b:
        if r > 200 and g < 100:
            return "Red"
        elif r > 150:
            return "Pink" if g > 100 else "Maroon"
        else:
            return "Brown"
    elif g > r and g > b:
        return "Green" if g > 150 else "Dark Green"
    elif b > r and b > g:
        return "Blue" if b > 150 else "Navy"
    elif r > 200 and g > 200:
        return "Yellow"
    else:
        return f"Color_RGB{tuple(rgb)}"

def merge_similar_colors(centers, props, indices):
    """Merge similar colors using Delta E"""
    raw_centers = [centers[i] for i in indices]
    swatches_rgb = [lab_opencv_to_rgb_tuple(c) for c in raw_centers]
    labcolor_centers = [lab_opencv_to_labcolor(c) for c in raw_centers]
    
    MERGE_DELTAE = 8.0
    final_centers, final_props, final_lab = [], [], []
    
    for labc, rgbc, prop in zip(labcolor_centers, swatches_rgb, props):
        chroma = math.sqrt(labc.lab_a**2 + labc.lab_b**2)
        
        # Don't drop important colors
        if prop < 0.02 and chroma < 2.0:
            continue
        
        merged = False
        for j, exist_lab in enumerate(final_lab):
            if delta_e_cie2000(labc, exist_lab) < MERGE_DELTAE:
                final_props[j] += prop
                merged = True
                break
        
        if not merged:
            final_lab.append(labc)
            final_centers.append(rgbc)
            final_props.append(prop)
    
    # Ensure we have colors
    if len(final_centers) == 0:
        final_centers = swatches_rgb[:5]
        final_props = props[:5].tolist()
    
    final_props = np.array(final_props, dtype=float)
    if final_props.sum() > 0:
        final_props /= final_props.sum()
    
    return final_centers, final_props

def classify_colors_hybrid(centers, proportions):
    """Classify colors with hex codes"""
    primary_colors = []
    secondary_colors = []
    
    for i, (color, percentage) in enumerate(zip(centers, proportions)):
        color_name = get_color_name_simple(color)
        hex_code = rgb_to_hex(color)
        
        color_info = {
            'rgb': tuple(color),
            'hex': hex_code,
            'name': color_name,
            'percentage': round(percentage * 100, 1)
        }
        
        if percentage >= 0.15:  # 15% threshold for primary
            primary_colors.append(color_info)
        elif percentage >= 0.05 and len(secondary_colors) < 2:  # 5% threshold for secondary
            secondary_colors.append(color_info)
    
    return primary_colors, secondary_colors

def plot_color_results(primary_colors, secondary_colors):
    """Plot color palette without names in bars"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    if primary_colors:
        colors_rgb = [np.array(c['rgb'])/255.0 for c in primary_colors]
        labels = [f"{c['percentage']}%" for c in primary_colors]
        
        bars1 = axes[0].bar(range(len(primary_colors)), [1]*len(primary_colors),
                           color=colors_rgb, width=0.7, edgecolor='black', linewidth=1)
        axes[0].set_xticks(range(len(primary_colors)))
        axes[0].set_xticklabels(labels, fontsize=11, fontweight='bold')
        axes[0].set_title('PRIMARY COLORS', fontsize=16, fontweight='bold', pad=20)
        axes[0].set_ylim(0, 1)
        axes[0].set_yticks([])
        
        for i, color in enumerate(primary_colors):
            text_color = 'white' if sum(color['rgb']) < 400 else 'black'
            axes[0].text(i, 0.5, f"{color['hex']}",
                        ha='center', va='center', fontweight='bold',
                        fontsize=10, color=text_color)
    else:
        axes[0].text(0.5, 0.5, 'No Primary Colors Found',
                    transform=axes[0].transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
    
    if secondary_colors:
        colors_rgb = [np.array(c['rgb'])/255.0 for c in secondary_colors]
        labels = [f"{c['percentage']}%" for c in secondary_colors]
        
        bars2 = axes[1].bar(range(len(secondary_colors)), [1]*len(secondary_colors),
                           color=colors_rgb, width=0.7, edgecolor='black', linewidth=1)
        axes[1].set_xticks(range(len(secondary_colors)))
        axes[1].set_xticklabels(labels, fontsize=11, fontweight='bold')
        axes[1].set_title('SECONDARY COLORS', fontsize=16, fontweight='bold', pad=20)
        axes[1].set_ylim(0, 1)
        axes[1].set_yticks([])
        
        for i, color in enumerate(secondary_colors):
            text_color = 'white' if sum(color['rgb']) < 400 else 'black'
            axes[1].text(i, 0.5, f"{color['hex']}",
                        ha='center', va='center', fontweight='bold',
                        fontsize=10, color=text_color)
    else:
        axes[1].text(0.5, 0.5, 'No Secondary Colors Found',
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=14, fontweight='bold')
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def analyze_garment_colors(image_path):
    """Hybrid analysis combining SAM and advanced color processing"""
    try:
        print("🔍 Advanced garment segmentation...")
        rgb, garment_mask = segment_garment_sam(image_path)
        print(f"   Segmentation complete. Found {np.sum(garment_mask):,} garment pixels.")
        
        print("🎨 Advanced color extraction...")
        centers, proportions, indices = extract_colors_advanced(rgb, garment_mask)
        
        print("🧬 Merging similar colors...")
        merged_centers, merged_props = merge_similar_colors(centers, proportions, indices)
        
        print("📊 Classifying colors...")
        primary_colors, secondary_colors = classify_colors_hybrid(merged_centers, merged_props)
        
        # Create visualizations
        masked_image = rgb.copy()
        masked_image[~garment_mask] = [0, 0, 0]
        
        # Display original and segmented images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(rgb)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(masked_image)
        axes[1].set_title('SAM Segmented Garment', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("🌈 Generating advanced color palette...")
        plot_color_results(primary_colors, secondary_colors)
        
        print("\n" + "="*70)
        print("            🎯 ADVANCED GARMENT COLOR ANALYSIS 🎯")
        print("="*70)
        
        print(f"\n🔴 PRIMARY COLORS ({len(primary_colors)} found):")
        print("-" * 50)
        if primary_colors:
            for i, color in enumerate(primary_colors, 1):
                print(f"   {i}. {color['name']:<20} {color['hex']:<10} RGB{str(color['rgb']):<20} {color['percentage']}%")
        else:
            print("   ❌ No primary colors detected")
        
        print(f"\n🟡 SECONDARY COLORS ({len(secondary_colors)} found):")
        print("-" * 50)
        if secondary_colors:
            for i, color in enumerate(secondary_colors, 1):
                print(f"   {i}. {color['name']:<20} {color['hex']:<10} RGB{str(color['rgb']):<20} {color['percentage']}%")
        else:
            print("   ❌ No secondary colors detected")
        
        print("="*70)
        
        return masked_image, primary_colors, secondary_colors
        
    except Exception as e:
        print(f"❌ Error in analyze_garment_colors: {e}")
        import traceback
        traceback.print_exc()
        return None, [], []