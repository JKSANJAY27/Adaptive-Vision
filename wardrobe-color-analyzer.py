# WARDROBE_ANALYZER_SOTA.py
# Patent-grade wardrobe color analysis with maximum accuracy
# - Multi-scale garment detection (Detectron2 + adaptive grid)
# - Enhanced clustering with color context awareness
# - Extensive color vocabulary with textile-specific naming
# - Robust to lighting variations and mixed fabrics

import warnings
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Optional deps - graceful fallbacks
HAS_CV = HAS_SK = HAS_SKIMG = HAS_D2 = HAS_TIMM = True

try:
    import cv2
except Exception:
    HAS_CV = False

try:
    from sklearn.cluster import KMeans, DBSCAN
except Exception:
    HAS_SK = False

try:
    from skimage import color as skcolor
except Exception:
    HAS_SKIMG = False

try:
    import torch
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
except Exception:
    HAS_D2 = False

try:
    import timm
except Exception:
    HAS_TIMM = False

def ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if img.mode != "RGB" else img

def to_bgr(a): return a[..., ::-1]
def rgb_to_hex(rgb):
    r,g,b = [int(x) for x in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

class WardrobeAnalyzerSOTA:
    def __init__(self, image_path: str, conf_thr: float = 0.25):
        self.image_path = image_path
        self.original_image = ensure_rgb(Image.open(image_path))
        self.W, self.H = self.original_image.size

        self.individual_garments = []
        self.garment_color_frequency = defaultdict(float)
        self.detected_colors = set()
        self.dominants = {}

        # Expanded color vocabulary for textile analysis
        self.color_categories = {
            "black":[(0,0,0),(12,12,12),(25,25,25),(35,35,35)],
            "white":[(255,255,255),(245,245,245),(235,235,235),(248,248,255)],
            "gray":[(128,128,128),(112,112,112),(160,160,160),(105,105,105),(169,169,169)],
            "red":[(220,20,60),(200,0,0),(255,69,0),(255,99,71),(178,34,34)],
            "maroon":[(128,0,0),(139,0,0),(115,6,6),(102,8,8)],
            "pink":[(255,105,180),(255,182,193),(255,192,203),(255,20,147),(219,112,147)],
            "orange":[(255,140,0),(255,165,0),(255,125,64),(255,69,0),(255,99,71)],
            "brown":[(139,69,19),(165,42,42),(150,75,0),(160,82,45),(205,133,63)],
            "yellow":[(255,215,0),(255,225,80),(255,255,0),(255,218,185),(240,230,140)],
            "green":[(34,139,34),(0,128,0),(46,139,87),(50,205,50),(124,252,0)],
            "teal":[(0,128,128),(32,160,160),(95,158,160),(72,209,204)],
            "blue":[(0,0,255),(70,130,180),(30,144,255),(100,149,237),(65,105,225)],
            "navy":[(0,0,128),(10,20,70),(25,25,112),(72,61,139)],
            "purple":[(128,0,128),(138,43,226),(147,112,219),(186,85,211)],
            "beige":[(245,245,220),(222,209,170),(250,235,215),(255,228,196)],
            "cream":[(255,253,208),(250,244,214),(255,255,240),(253,245,230)]
        }

        self._init_segmenter(conf_thr)

    def _init_segmenter(self, conf_thr: float = 0.25):
        self.segmenter = None
        if not HAS_D2:
            print("Detectron2 not available; using multi-scale grid analysis.")
            return
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = conf_thr
        cfg.MODEL.DEVICE = "cuda" if ("torch" in globals() and torch.cuda.is_available()) else "cpu"
        self.segmenter = DefaultPredictor(cfg)
        print("Segmenter initialized (Mask R-CNN R50-FPN).")

    # ---------- Enhanced Detection ----------
    def detect_garments(self):
        # Combine Detectron2 with multi-scale grid for comprehensive coverage
        garments_d2 = self._detect_with_detectron2() if self.segmenter and HAS_CV else []
        garments_grid = self._detect_with_adaptive_grid()
        
        # Merge and deduplicate
        all_garments = garments_d2 + garments_grid
        self.individual_garments = self._deduplicate_garments(all_garments)
        
        print(f"Detected {len(self.individual_garments)} garment regions for analysis.")
        return len(self.individual_garments)

    def _detect_with_detectron2(self):
        img_bgr = to_bgr(np.array(self.original_image))
        outputs = self.segmenter(img_bgr)
        inst = outputs["instances"].to("cpu")
        
        garments = []
        if inst.has("pred_masks"):
            masks = inst.pred_masks.numpy()
            scores = inst.scores.numpy()
            boxes = inst.pred_boxes.tensor.numpy()
            
            H, W = img_bgr.shape[:2]
            for gid, (mask, sc, box) in enumerate(zip(masks, scores, boxes)):
                if sc < 0.25: continue
                area = int(mask.sum())
                if area < 0.001 * H * W: continue  # More permissive for individual items
                
                pixels = img_bgr[mask.astype(bool)][:, ::-1]  # to RGB
                if len(pixels) < 100: continue
                
                x1, y1, x2, y2 = map(int, box)
                garments.append({
                    "id": f"D2_{gid}",
                    "bbox": (x1, y1, x2-x1, y2-y1),
                    "pixels": pixels,
                    "area": area,
                    "confidence": float(sc),
                    "method": "detectron2"
                })
        
        return garments

    def _detect_with_adaptive_grid(self):
        # Multi-scale grid to catch individual garments
        img = np.array(self.original_image)
        H, W = img.shape[:2]
        
        garments = []
        gid = 0
        
        # Horizontal strips (for hanging clothes)
        for rows in [4, 6, 8]:
            strip_h = H // rows
            for r in range(rows):
                y1 = r * strip_h
                y2 = H if r == rows - 1 else (r + 1) * strip_h
                
                crop = img[y1:y2, :, :]
                pixels = crop.reshape(-1, 3)
                if len(pixels) < 500: continue
                
                # Check if this region has sufficient color variation (likely clothing)
                if self._has_clothing_characteristics(pixels):
                    garments.append({
                        "id": f"GRID_H_{gid}",
                        "bbox": (0, y1, W, y2-y1),
                        "pixels": pixels,
                        "area": int((y2-y1) * W),
                        "confidence": 0.7,
                        "method": "grid_horizontal"
                    })
                    gid += 1
        
        # Vertical strips (for folded/organized clothes)
        for cols in [6, 8, 12]:
            strip_w = W // cols
            for c in range(cols):
                x1 = c * strip_w
                x2 = W if c == cols - 1 else (c + 1) * strip_w
                
                crop = img[:, x1:x2, :]
                pixels = crop.reshape(-1, 3)
                if len(pixels) < 500: continue
                
                if self._has_clothing_characteristics(pixels):
                    garments.append({
                        "id": f"GRID_V_{gid}",
                        "bbox": (x1, 0, x2-x1, H),
                        "pixels": pixels,
                        "area": int(H * (x2-x1)),
                        "confidence": 0.6,
                        "method": "grid_vertical"
                    })
                    gid += 1
        
        return garments

    def _has_clothing_characteristics(self, pixels):
        # Simple heuristic: clothing regions have moderate color variation and aren't predominantly background
        if len(pixels) < 100:
            return False
            
        # Check color diversity
        std = np.std(pixels, axis=0).mean()
        if std < 15:  # Too uniform (likely background)
            return False
            
        # Check if not predominantly very bright or very dark (likely background)
        mean_brightness = np.mean(pixels)
        if mean_brightness > 240 or mean_brightness < 15:
            return False
            
        return True

    def _deduplicate_garments(self, garments):
        # Remove heavily overlapping regions, keeping higher confidence ones
        if not garments:
            return []
            
        # Sort by confidence descending
        garments = sorted(garments, key=lambda x: x['confidence'], reverse=True)
        
        kept = []
        for g in garments:
            overlap = False
            for k in kept:
                if self._bbox_overlap_ratio(g['bbox'], k['bbox']) > 0.7:
                    overlap = True
                    break
            if not overlap:
                kept.append(g)
                
        return kept[:20]  # Limit to prevent excessive regions

    def _bbox_overlap_ratio(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
            
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    # ---------- Enhanced Color Processing ----------
    def _rgb_to_lab(self, rgb_uint8):
        if not HAS_SKIMG:
            # Simple approximation
            return (rgb_uint8.astype(np.float32) / 255.0) * 100.0
        rgb = (rgb_uint8.astype(np.float32)/255.0).reshape(-1,1,3)
        lab = skcolor.rgb2lab(rgb).reshape(-1,3)
        return lab

    def _cluster_dominant_enhanced(self, pixels_rgb):
        """Enhanced clustering with better parameter adaptation"""
        if not HAS_SK:
            # Fallback: simple color quantization
            return self._simple_color_quantization(pixels_rgb)
            
        N = len(pixels_rgb)
        if N < 200:
            return self._simple_color_quantization(pixels_rgb)
            
        lab = self._rgb_to_lab(pixels_rgb)
        
        # Adaptive k based on color complexity
        color_std = np.std(pixels_rgb, axis=0).mean()
        if color_std > 60:  # High variation
            k = min(8, max(3, N // 8000))
        elif color_std > 30:  # Medium variation
            k = min(6, max(2, N // 12000))
        else:  # Low variation
            k = min(4, max(2, N // 15000))
        
        # Initial clustering in LAB space
        km = KMeans(n_clusters=k, n_init=15, random_state=42, max_iter=300)
        labels = km.fit_predict(lab)
        centers = km.cluster_centers_
        
        # Filter by minimum cluster size (adaptive threshold)
        counts = np.bincount(labels)
        min_size = max(0.03, 100.0 / N)  # At least 3% or 100 pixels
        keep = counts / N >= min_size
        
        if not np.any(keep):  # Emergency fallback
            keep[np.argmax(counts)] = True
            
        centers = centers[keep]
        
        # Enhanced merging with distance-based clustering
        if len(centers) > 1:
            # Use adaptive eps based on color spread
            color_range = np.ptp(lab, axis=0).mean()
            eps = min(12.0, max(6.0, color_range * 0.15))
            
            db = DBSCAN(eps=eps, min_samples=1, metric="euclidean").fit(centers)
            unique_labels = np.unique(db.labels_)
            
            merged_centers = []
            for label in unique_labels:
                cluster_centers = centers[db.labels_ == label]
                merged_centers.append(cluster_centers.mean(axis=0))
            
            centers = np.array(merged_centers)
        
        # Convert back to RGB
        centers_rgb = self._lab_to_rgb(centers)
        
        # Recalculate frequencies based on final centers
        distances = np.linalg.norm(lab[:, None, :] - centers[None, :, :], axis=2)
        final_labels = distances.argmin(axis=1)
        frequencies = np.bincount(final_labels, minlength=len(centers)) / N
        
        return centers_rgb, frequencies

    def _simple_color_quantization(self, pixels_rgb):
        """Fallback when sklearn not available"""
        # Simple k-means-like approach
        unique_colors = np.unique(pixels_rgb.reshape(-1, 3), axis=0)
        if len(unique_colors) <= 5:
            colors, counts = np.unique(pixels_rgb.reshape(-1, 3), axis=0, return_counts=True)
            frequencies = counts / counts.sum()
            return colors, frequencies
        
        # Sample-based approach
        n_colors = min(5, len(unique_colors))
        indices = np.random.choice(len(unique_colors), n_colors, replace=False)
        centers = unique_colors[indices]
        
        # Assign pixels to nearest center
        distances = np.linalg.norm(pixels_rgb[:, None, :] - centers[None, :, :], axis=2)
        labels = distances.argmin(axis=1)
        frequencies = np.bincount(labels, minlength=len(centers)) / len(pixels_rgb)
        
        return centers, frequencies

    def _lab_to_rgb(self, lab):
        if not HAS_SKIMG:
            x = np.clip(lab / 100.0, 0, 1)
            return (x * 255).astype(np.uint8)
        rgb = skcolor.lab2rgb(lab.reshape(-1,1,3)).reshape(-1,3)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    def _delta_e_enhanced(self, lab1, lab2):
        """Enhanced perceptual distance with lighting adaptation"""
        a = np.array(lab1, dtype=np.float32).ravel()
        b = np.array(lab2, dtype=np.float32).ravel()
        
        # Standard LAB Euclidean with perceptual weighting
        # Weight L less (brightness less important for clothing color ID)
        weights = np.array([0.7, 1.2, 1.1])  # [L, a, b] weights
        diff = (a - b) * weights
        
        return float(np.linalg.norm(diff))

    def _rgb_to_lab_corrected(self, rgb_tuple):
        """RGB to LAB with lighting correction for indoor environments"""
        lab = self._rgb_to_lab(np.array([rgb_tuple], dtype=np.uint8))[0]
        
        # Wardrobe-specific lighting corrections
        lab[1] *= 0.90  # Reduce green-red axis (common indoor shift)
        lab[2] *= 0.88  # Reduce blue-yellow axis (warm indoor lighting)
        
        return lab

    def classify_color_enhanced(self, rgb_tuple):
        """Enhanced color classification with context awareness"""
        r, g, b = rgb_tuple
        
        # Enhanced grayscale detection
        color_range = max(r, g, b) - min(r, g, b)
        if color_range < 20:  # More lenient grayscale threshold
            brightness = int(round((r + g + b) / 3))
            if brightness > 225: return "white"
            if brightness < 35: return "black"
            return "gray"
        
        # Main color matching with enhanced distance
        best_match = ("unknown", 1e9)
        test_lab = self._rgb_to_lab_corrected(rgb_tuple)
        
        for color_name, variants in self.color_categories.items():
            min_distance = 1e9
            for variant_rgb in variants:
                ref_lab = self._rgb_to_lab_corrected(variant_rgb)
                distance = self._delta_e_enhanced(test_lab, ref_lab)
                min_distance = min(min_distance, distance)
            
            if min_distance < best_match[1]:
                best_match = (color_name, min_distance)
        
        # Enhanced acceptance threshold with context
        base_threshold = 25.0
        
        # Adjust threshold based on color characteristics
        if color_range > 80:  # High saturation colors
            threshold = base_threshold * 0.8
        elif color_range < 40:  # Low saturation colors
            threshold = base_threshold * 1.2
        else:
            threshold = base_threshold
        
        if best_match[1] <= threshold:
            return best_match[0]
        
        return "unknown"

    def extract_colors_enhanced(self):
        """Enhanced color extraction with better sampling and filtering"""
        if not self.individual_garments:
            print("No garments detected for color analysis.")
            return {}
        
        all_colors = {}
        color_id = 0
        
        for garment in self.individual_garments:
            pixels = garment["pixels"]
            N = len(pixels)
            
            if N < 200:
                continue
            
            # Enhanced sampling strategy
            if N > 50000:
                # For very large regions, use stratified sampling
                sample_size = min(50000, N)
                indices = np.random.choice(N, sample_size, replace=False)
                sample_pixels = pixels[indices]
            else:
                sample_pixels = pixels
            
            # Get dominant colors
            centers_rgb, frequencies = self._cluster_dominant_enhanced(sample_pixels)
            
            for center_rgb, freq in zip(centers_rgb, frequencies):
                # More lenient frequency threshold for individual garments
                min_freq = 0.05 if garment["method"] == "detectron2" else 0.08
                
                if freq < min_freq:
                    continue
                
                # Classify color
                color_name = self.classify_color_enhanced(tuple(int(v) for v in center_rgb))
                
                # Store color information
                color_key = f"C{color_id}_{garment['id']}"
                all_colors[color_key] = {
                    "rgb": tuple(int(v) for v in center_rgb),
                    "hex": rgb_to_hex(center_rgb),
                    "category": color_name,
                    "frequency": float(freq),
                    "garment_id": garment["id"],
                    "detection_method": garment["method"]
                }
                
                # Update global frequency (weighted by garment confidence)
                weight = garment.get("confidence", 0.5)
                weighted_freq = freq * weight
                self.garment_color_frequency[color_name] += weighted_freq
                
                if color_name != "unknown":
                    self.detected_colors.add(color_name)
                
                color_id += 1
        
        self.dominants = all_colors
        print(f"Extracted {len(all_colors)} color instances across {len(self.individual_garments)} garment regions.")
        return all_colors

    def analyze(self, visualize=True):
        """Main analysis pipeline with enhanced accuracy"""
        print("Starting enhanced wardrobe analysis...")
        
        # Detection phase
        garment_count = self.detect_garments()
        
        # Color extraction phase
        self.extract_colors_enhanced()
        
        # Normalize color frequencies
        total_freq = sum(self.garment_color_frequency.values())
        if total_freq > 0:
            normalized_freq = {
                color: freq / total_freq 
                for color, freq in self.garment_color_frequency.items()
                if freq > 0
            }
        else:
            normalized_freq = {}
        
        results = {
            "garment_count": garment_count,
            "current_colors": sorted(list(self.detected_colors)),
            "color_distribution": dict(sorted(normalized_freq.items(), key=lambda x: x[1], reverse=True)),
            "dominants": self.dominants,
            "detection_summary": {
                "total_regions": len(self.individual_garments),
                "detectron2_regions": len([g for g in self.individual_garments if g["method"] == "detectron2"]),
                "grid_regions": len([g for g in self.individual_garments if "grid" in g["method"]])
            }
        }
        
        if visualize:
            self._plot_results_enhanced(results)
        
        return results

    def _plot_results_enhanced(self, results):
        """Enhanced visualization with complete color representation"""
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.0, 0.8], hspace=0.3, wspace=0.3)
        
        # Main image with detection boxes
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.original_image)
        ax1.set_title(f"Detected Regions ({results['garment_count']})", fontsize=14, fontweight='bold')
        
        # Draw detection boxes with different colors for different methods
        method_colors = {"detectron2": "red", "grid_horizontal": "blue", "grid_vertical": "green"}
        for garment in self.individual_garments:
            x, y, w, h = garment["bbox"]
            color = method_colors.get(garment["method"], "orange")
            ax1.add_patch(plt.Rectangle((x, y), w, h, fill=False, color=color, linewidth=1.5, alpha=0.7))
        
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Enhanced pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        
        color_dist = results["color_distribution"]
        if color_dist:
            # Filter out very small percentages for cleaner display
            filtered_dist = {k: v for k, v in color_dist.items() if v >= 0.02}  # At least 2%
            
            if not filtered_dist:
                filtered_dist = dict(list(color_dist.items())[:5])  # Top 5 if all are small
            
            labels = list(filtered_dist.keys())
            sizes = list(filtered_dist.values())
            
            # Generate colors for pie chart
            pie_colors = []
            for color_name in labels:
                if color_name == "unknown":
                    pie_colors.append("#888888")
                elif color_name in ["black", "white", "gray"]:
                    color_map = {"black": "#1a1a1a", "white": "#f5f5f5", "gray": "#808080"}
                    pie_colors.append(color_map[color_name])
                else:
                    # Use representative color from category
                    rep_rgb = self.color_categories.get(color_name, [(128, 128, 128)])[0]
                    pie_colors.append(rgb_to_hex(rep_rgb))
            
            wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                              colors=pie_colors, startangle=90)
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
        else:
            ax2.text(0.5, 0.5, "No colors detected", ha='center', va='center', 
                    fontsize=12, transform=ax2.transAxes)
        
        ax2.set_title("Color Distribution", fontsize=14, fontweight='bold')
        
        # FIXED: Color breakdown bar chart - Show ALL detected colors
        ax3 = fig.add_subplot(gs[1, :])
        
        if color_dist:
            # Show ALL colors that were detected, not just top 10
            all_colors = dict(color_dist.items())
            x_pos = range(len(all_colors))
            
            # Generate consistent colors for all bars
            bar_colors = []
            for color_name in all_colors.keys():
                if color_name == "unknown":
                    bar_colors.append("#888888")
                elif color_name in ["black", "white", "gray"]:
                    color_map = {"black": "#1a1a1a", "white": "#f5f5f5", "gray": "#808080"}
                    bar_colors.append(color_map[color_name])
                else:
                    # Use representative color from category
                    rep_rgb = self.color_categories.get(color_name, [(128, 128, 128)])[0]
                    bar_colors.append(rgb_to_hex(rep_rgb))
            
            bars = ax3.bar(x_pos, all_colors.values(), color=bar_colors)
            
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(all_colors.keys(), rotation=45, ha='right')
            ax3.set_ylabel('Relative Frequency')
            ax3.set_title('Complete Color Breakdown - All Detected Colors', fontsize=12, fontweight='bold')
            
            # Add value labels on bars (show percentage)
            for bar, (color_name, value) in zip(bars, all_colors.items()):
                height = bar.get_height()
                if height > 0.01:  # Only label if > 1%
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                            f'{value:.1%}', ha='center', va='bottom', fontsize=8, rotation=0)
        
        # Statistics summary
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_text = f"""
Analysis Summary:
• Total Garment Regions: {results['detection_summary']['total_regions']}
• Detectron2 Instances: {results['detection_summary']['detectron2_regions']}
• Grid Regions: {results['detection_summary']['grid_regions']}
• Distinct Colors Found: {len(results['current_colors'])}
• Colors: {', '.join(results['current_colors']) if results['current_colors'] else 'None detected'}
        """
        
        ax4.text(0.05, 0.95, stats_text.strip(), transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()

def analyze_wardrobe(image_path: str, visualize: bool = True):
    """Main entry point for wardrobe analysis"""
    analyzer = WardrobeAnalyzerSOTA(image_path=image_path)
    return analyzer.analyze(visualize=visualize)
