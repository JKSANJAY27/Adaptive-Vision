
"""
FIXED Enhanced Professional Wardrobe Color Analysis System
=========================================================

Fixed version that properly displays visualizations without closing immediately.
All matplotlib compatibility issues resolved.

Patent Reference: "Wardrobe Colour Diversity and Colour Gamut Profile Based Garment 
Match Finding Enabler for Colourblind Individuals"

Version: 2.1 (FIXED Display Version)
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from sklearn.cluster import KMeans, MeanShift
from collections import defaultdict, Counter
import colorsys
import json
from datetime import datetime
from PIL import Image, ImageEnhance
from scipy.spatial.distance import euclidean
import pandas as pd
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

def convert_numpy(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy(i) for i in obj)
    else:
        return obj

class FixedProfessionalWardrobeAnalyzer:
    """
    FIXED Professional Wardrobe Analyzer with resolved display issues
    """

    def __init__(self):
        # Enhanced comprehensive color categorization system
        self.color_categories = {
            'red': [(220, 20, 60), (255, 0, 0), (178, 34, 34), (165, 42, 42), (139, 0, 0), (205, 92, 92)],
            'pink': [(255, 192, 203), (255, 20, 147), (255, 105, 180), (219, 112, 147), (255, 182, 193)],
            'blue': [(0, 0, 255), (30, 144, 255), (70, 130, 180), (100, 149, 237), (65, 105, 225), (0, 191, 255)],
            'lightblue': [(173, 216, 230), (135, 206, 235), (176, 196, 222), (230, 230, 250), (135, 206, 250)],
            'green': [(0, 128, 0), (34, 139, 34), (50, 205, 50), (46, 139, 87), (144, 238, 144), (0, 255, 127)],
            'yellow': [(255, 255, 0), (255, 215, 0), (255, 255, 224), (240, 230, 140), (255, 250, 205)],
            'orange': [(255, 165, 0), (255, 140, 0), (255, 127, 80), (255, 99, 71), (255, 69, 0)],
            'purple': [(128, 0, 128), (147, 112, 219), (138, 43, 226), (186, 85, 211), (148, 0, 211)],
            'brown': [(165, 42, 42), (139, 69, 19), (160, 82, 45), (205, 133, 63), (210, 180, 140)],
            'black': [(0, 0, 0), (25, 25, 25), (64, 64, 64), (105, 105, 105), (47, 79, 79)],
            'white': [(255, 255, 255), (248, 248, 255), (245, 245, 245), (250, 250, 250), (255, 250, 240)],
            'gray': [(128, 128, 128), (169, 169, 169), (192, 192, 192), (211, 211, 211), (119, 136, 153)],
            'navy': [(0, 0, 128), (25, 25, 112), (72, 61, 139), (47, 79, 79), (75, 0, 130)],
            'beige': [(245, 245, 220), (255, 228, 196), (222, 184, 135), (210, 180, 140), (238, 203, 173)],
            'maroon': [(128, 0, 0), (139, 0, 0), (165, 42, 42), (220, 20, 60), (176, 48, 96)]
        }

        # Enhanced wardrobe color categories for diversity analysis
        self.essential_colors = ['black', 'white', 'navy', 'blue', 'gray', 'red', 'brown']
        self.accent_colors = ['pink', 'green', 'yellow', 'orange', 'purple', 'lightblue']

        # Analysis results storage
        self.garment_detections = []
        self.color_analysis = {}
        self.diversity_metrics = {}
        self.analysis_stats = {}

    def preprocess_image(self, image_path):
        """Enhanced image preprocessing with advanced enhancement algorithms"""
        try:
            if isinstance(image_path, str):
                pil_image = Image.open(image_path).convert('RGB')
                print(f"✓ Image loaded: {pil_image.size}")
            else:
                pil_image = image_path.convert('RGB')

            # Professional image enhancement pipeline
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Sharpness(enhanced)
            sharpened = enhancer.enhance(1.2)
            enhancer = ImageEnhance.Color(sharpened)
            color_enhanced = enhancer.enhance(1.1)

            return np.array(color_enhanced)
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            return None

    def advanced_garment_detection(self, image):
        """Enhanced multi-region garment detection with improved accuracy"""
        try:
            height, width, _ = image.shape
            print(f"🔍 Analyzing image: {width}x{height} pixels")

            garment_regions = []

            # Enhanced grid-based analysis
            num_cols = 10
            num_rows = 4

            region_width = width // num_cols
            region_height = height // num_rows

            detected_regions = 0

            for i in range(num_cols):
                x = int(i * region_width * 0.8)
                w = min(int(region_width * 1.4), width - x)

                for j in range(num_rows):
                    y = int(j * region_height * 0.7)
                    h = min(int(region_height * 1.5), height - y)

                    if x + w <= width and y + h <= height:
                        region_roi = image[y:y+h, x:x+w]

                        if self.has_significant_content(region_roi):
                            refined_bbox = self.refine_bounding_box(image, (x, y, w, h))

                            if refined_bbox:
                                rx, ry, rw, rh = refined_bbox
                                if rw > 35 and rh > 50:
                                    confidence = self.calculate_region_confidence(region_roi)
                                    if confidence > 0.6:
                                        garment_regions.append({
                                            'bbox': refined_bbox,
                                            'area': rw * rh,
                                            'confidence': confidence
                                        })
                                        detected_regions += 1

            print(f"🎯 Initial regions detected: {detected_regions}")

            garment_regions = self.remove_overlapping_regions(garment_regions)
            garment_regions.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)

            final_regions = garment_regions[:20]
            print(f"✅ Final garment regions: {len(final_regions)}")

            return final_regions

        except Exception as e:
            print(f"❌ Detection error: {e}")
            return []

    def has_significant_content(self, region):
        """Enhanced content detection with multiple criteria"""
        if region.size == 0:
            return False

        flat_region = region.reshape(-1, 3)
        color_std = np.std(flat_region, axis=0).mean()

        gray = np.mean(region, axis=2)
        dx = np.gradient(gray, axis=1)
        dy = np.gradient(gray, axis=0)
        edges = np.sqrt(dx**2 + dy**2)
        edge_content = np.sum(edges) / edges.size

        texture_variance = np.var(gray)

        return (color_std > 12 or edge_content > 0.015 or texture_variance > 100)

    def calculate_region_confidence(self, region):
        """Calculate confidence score for detected region"""
        try:
            gray = np.mean(region, axis=2)

            edge_strength = np.std(np.gradient(gray))
            color_diversity = np.std(region.reshape(-1, 3), axis=0).mean()
            texture_score = np.var(gray)

            confidence = (
                min(edge_strength / 10, 1) * 0.3 +
                min(color_diversity / 50, 1) * 0.4 +
                min(texture_score / 1000, 1) * 0.3
            )

            return min(confidence, 1.0)
        except:
            return 0.5

    def refine_bounding_box(self, image, bbox):
        """Enhanced bounding box refinement"""
        try:
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]

            gray_roi = np.mean(roi, axis=2)
            mean_val = np.mean(gray_roi)
            std_val = np.std(gray_roi)
            threshold = mean_val - std_val * 0.5

            content_mask = gray_roi < threshold

            # Simple morphological operations
            from scipy import ndimage
            content_mask = ndimage.binary_closing(content_mask, structure=np.ones((3,3)))
            content_mask = ndimage.binary_opening(content_mask, structure=np.ones((2,2)))

            rows = np.any(content_mask, axis=1)
            cols = np.any(content_mask, axis=0)

            if not np.any(rows) or not np.any(cols):
                return bbox

            row_indices = np.where(rows)[0]
            col_indices = np.where(cols)[0]

            if len(row_indices) > 0 and len(col_indices) > 0:
                rmin, rmax = row_indices[0], row_indices[-1]
                cmin, cmax = col_indices[0], col_indices[-1]

                padding = 5
                rmin = max(0, rmin - padding)
                rmax = min(h - 1, rmax + padding)
                cmin = max(0, cmin - padding)
                cmax = min(w - 1, cmax + padding)

                new_x = x + cmin
                new_y = y + rmin
                new_w = cmax - cmin + 1
                new_h = rmax - rmin + 1

                return (new_x, new_y, new_w, new_h)

        except:
            pass

        return bbox

    def remove_overlapping_regions(self, regions):
        """Enhanced overlap removal"""
        if len(regions) <= 1:
            return regions

        regions.sort(key=lambda x: (x['confidence'], x['area']), reverse=True)

        filtered_regions = []
        for region in regions:
            is_duplicate = False

            for existing in filtered_regions:
                overlap = self.calculate_overlap(region['bbox'], existing['bbox'])
                if overlap > 0.4:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered_regions.append(region)

        return filtered_regions

    def calculate_overlap(self, bbox1, bbox2):
        """Enhanced IoU calculation"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)

        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0

        return 0

    def extract_garment_colors(self, image, region_info):
        """Enhanced color extraction with improved clustering"""
        try:
            x, y, w, h = region_info['bbox']

            y1, y2 = max(0, y), min(image.shape[0], y + h)
            x1, x2 = max(0, x), min(image.shape[1], x + w)

            roi = image[y1:y2, x1:x2]

            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                return None, []

            pixels = roi.reshape(-1, 3)

            pixel_brightness = np.mean(pixels, axis=1)
            q1, q3 = np.percentile(pixel_brightness, [20, 80])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask = (pixel_brightness >= lower_bound) & (pixel_brightness <= upper_bound)
            clean_pixels = pixels[mask] if np.any(mask) and np.sum(mask) > 50 else pixels

            if len(clean_pixels) < 50:
                return None, []

            n_colors = min(6, max(3, len(clean_pixels) // 80))

            # Use KMeans for better compatibility
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(clean_pixels)
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_

            color_info = []
            total_pixels = len(labels)

            for i, color in enumerate(colors):
                count = np.sum(labels == i)
                proportion = count / total_pixels

                if proportion > 0.03:
                    color_name = self.classify_color_enhanced(color)
                    confidence = self.calculate_color_confidence(color, clean_pixels[labels == i])

                    if confidence > 0.6:
                        color_info.append({
                            'rgb': tuple(np.clip(color.astype(int), 0, 255)),
                            'proportion': proportion,
                            'name': color_name,
                            'confidence': confidence
                        })

            color_info.sort(key=lambda x: x['proportion'], reverse=True)

            return roi, color_info[:4]

        except Exception as e:
            print(f"⚠️  Color extraction error: {e}")
            return None, []

    def classify_color_enhanced(self, rgb_color):
        """Enhanced color classification with improved accuracy"""
        r, g, b = np.clip(rgb_color, 0, 255)

        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)

        if s < 0.08 or v < 0.12:
            if v < 0.2:
                return 'black'
            elif v > 0.88:
                return 'white'
            else:
                return 'gray'

        if 0.05 < s < 0.4 and 0.6 < v < 0.95:
            if 0.08 < h < 0.17:
                return 'beige'

        min_distance = float('inf')
        closest_color = 'gray'

        for color_name, color_values in self.color_categories.items():
            for color_rgb in color_values:
                distance = np.sqrt(
                    ((r - color_rgb[0]) * 0.3)**2 + 
                    ((g - color_rgb[1]) * 0.59)**2 + 
                    ((b - color_rgb[2]) * 0.11)**2
                )

                if distance < min_distance:
                    min_distance = distance
                    closest_color = color_name

        return closest_color

    def calculate_color_confidence(self, color, pixels):
        """Calculate confidence score for color classification"""
        try:
            if len(pixels) == 0:
                return 0.5

            distances = [euclidean(color, pixel) for pixel in pixels[:100]]
            consistency = 1.0 - (np.mean(distances) / 255.0)

            h, s, v = colorsys.rgb_to_hsv(color[0]/255, color[1]/255, color[2]/255)
            saturation_score = s
            brightness_score = 1.0 - abs(v - 0.5) * 2

            confidence = (consistency * 0.5 + saturation_score * 0.3 + brightness_score * 0.2)

            return min(max(confidence, 0.1), 1.0)
        except:
            return 0.7

    def calculate_diversity_score(self, color_counts):
        """Enhanced Shannon entropy-based diversity calculation"""
        if not color_counts or sum(color_counts.values()) == 0:
            return 0

        total = sum(color_counts.values())
        shannon_index = 0

        for count in color_counts.values():
            if count > 0:
                proportion = count / total
                shannon_index -= proportion * np.log(proportion)

        max_possible = np.log(len(self.color_categories))
        base_score = (shannon_index / max_possible) * 100 if max_possible > 0 else 0

        variety_bonus = min(len(color_counts) * 2, 10)
        diversity_score = min(100, base_score + variety_bonus)

        return diversity_score

    def analyze_wardrobe(self, image_path):
        """Enhanced complete wardrobe analysis pipeline"""
        print("🚀 Starting FIXED Professional Wardrobe Analysis...")
        print("=" * 60)

        image = self.preprocess_image(image_path)
        if image is None:
            print("❌ Failed to load image")
            return False

        print("\n🔎 Phase 1: Advanced Garment Detection...")
        garment_regions = self.advanced_garment_detection(image)

        if not garment_regions:
            print("❌ No garments detected")
            return False

        print("\n🎨 Phase 2: Color Analysis & Classification...")
        color_frequency = defaultdict(float)
        garment_details = []
        successful_analyses = 0

        for i, region in enumerate(garment_regions):
            print(f"   Analyzing garment {i+1}/{len(garment_regions)}...", end=" ")

            roi, colors = self.extract_garment_colors(image, region)

            if colors:
                garment_id = f"G{i+1}"

                garment_info = {
                    'id': garment_id,
                    'bbox': region['bbox'],
                    'colors': colors,
                    'primary_color': colors[0]['name'] if colors else 'unknown',
                    'confidence': region['confidence'],
                    'color_count': len(colors)
                }

                garment_details.append(garment_info)
                successful_analyses += 1

                for color_data in colors:
                    weight = color_data['proportion'] * color_data.get('confidence', 1.0)
                    color_frequency[color_data['name']] += weight

                print("✓")
            else:
                print("⚠️")

        print(f"\n📊 Phase 3: Calculating Diversity Metrics...")

        diversity_score = self.calculate_diversity_score(color_frequency)

        essential_present = sum(1 for color in self.essential_colors if color in color_frequency)
        essential_coverage = (essential_present / len(self.essential_colors)) * 100

        accent_present = sum(1 for color in self.accent_colors if color in color_frequency)
        accent_coverage = (accent_present / len(self.accent_colors)) * 100

        color_balance = min(essential_coverage, accent_coverage)
        health_score = (diversity_score * 0.4 + essential_coverage * 0.3 + accent_coverage * 0.3)

        self.garment_detections = garment_details
        self.color_analysis = dict(color_frequency)
        self.diversity_metrics = {
            'diversity_score': diversity_score,
            'total_garments': len(garment_details),
            'unique_colors': len(color_frequency),
            'essential_coverage': essential_coverage,
            'accent_coverage': accent_coverage,
            'color_balance': color_balance,
            'health_score': health_score,
            'successful_analyses': successful_analyses,
            'detection_rate': (successful_analyses / len(garment_regions)) * 100
        }

        self.analysis_stats = {
            'image_dimensions': f"{image.shape[1]}x{image.shape[0]}",
            'total_regions_detected': len(garment_regions),
            'successful_color_analyses': successful_analyses,
            'average_colors_per_garment': np.mean([len(g['colors']) for g in garment_details]),
            'most_common_color': max(color_frequency.keys(), key=color_frequency.get) if color_frequency else 'None'
        }

        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Diversity Score: {diversity_score:.1f}/100")
        print(f"🏥 Health Score: {health_score:.1f}/100")
        print(f"👔 Garments Analyzed: {successful_analyses}/{len(garment_regions)}")
        print(f"🎨 Unique Colors: {len(color_frequency)}")
        print(f"⚡ Essential Coverage: {essential_coverage:.1f}%")
        print(f"✨ Accent Coverage: {accent_coverage:.1f}%")

        return True

    def create_fixed_visualization(self, image_path, save_path="fixed_professional_analysis.png"):
        """Create FIXED visualization that displays properly"""
        if not self.garment_detections:
            print("❌ No analysis data available for visualization")
            return None

        image = self.preprocess_image(image_path)
        if image is None:
            return None

        print("🎨 Creating FIXED professional visualization...")

        # Create figure with FIXED settings
        plt.rcParams.update({
            'figure.max_open_warning': 0,
            'axes.formatter.use_mathtext': True,
            'font.size': 10
        })

        fig = plt.figure(figsize=(18, 11), facecolor='white', dpi=100)
        gs = GridSpec(3, 4, height_ratios=[2.5, 1, 1], width_ratios=[2, 1, 1, 1], 
                      hspace=0.25, wspace=0.25)

        # Main image with bounding boxes
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.imshow(image)

        colors_for_boxes = plt.cm.tab20(np.linspace(0, 1, len(self.garment_detections)))

        for i, garment in enumerate(self.garment_detections):
            x, y, w, h = garment['bbox']

            confidence = garment.get('confidence', 0.8)
            line_width = 2 + confidence * 3

            rect = patches.Rectangle(
                (x, y), w, h, 
                linewidth=line_width, 
                edgecolor=colors_for_boxes[i], 
                facecolor='none',
                alpha=0.8
            )
            ax1.add_patch(rect)

            color_count = garment.get('color_count', 1)
            label_text = f"{garment['id']}\n{garment['primary_color'].title()}\n({color_count} colors)"

            label_y = max(0, y-15)
            ax1.text(
                x+3, label_y, 
                label_text,
                fontsize=9, 
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor=colors_for_boxes[i], 
                         alpha=0.9,
                         edgecolor='white',
                         linewidth=1),
                verticalalignment='bottom',
                color='white'
            )

        ax1.set_title("🔍 Individual Garment Detection", 
                     fontsize=14, fontweight='bold', pad=15)
        ax1.axis('off')

        # Color distribution pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        if self.color_analysis:
            colors_list = list(self.color_analysis.keys())
            sizes = list(self.color_analysis.values())

            color_map = {
                'red': '#DC143C', 'pink': '#FF1493', 'blue': '#4169E1',
                'green': '#32CD32', 'yellow': '#FFD700', 'orange': '#FF8C00',
                'purple': '#9370DB', 'brown': '#8B4513', 'black': '#2F2F2F',
                'white': '#F8F8FF', 'gray': '#708090', 'navy': '#191970',
                'beige': '#F5E6D3', 'maroon': '#800000', 'lightblue': '#87CEEB'
            }

            pie_colors = [color_map.get(color, '#CCCCCC') for color in colors_list]

            wedges, texts, autotexts = ax2.pie(
                sizes, 
                labels=colors_list, 
                colors=pie_colors,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 8, 'fontweight': 'bold'}
            )

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        ax2.set_title("🎨 Color Distribution", fontsize=12, fontweight='bold', pad=10)

        # Metrics panel
        ax3 = fig.add_subplot(gs[0, 3])
        ax3.axis('off')

        diversity_score = self.diversity_metrics['diversity_score']
        health_score = self.diversity_metrics['health_score']

        metrics_text = f"""📊 ANALYSIS SUMMARY

🎯 Diversity: {diversity_score:.1f}/100
🏥 Health: {health_score:.1f}/100
👔 Garments: {self.diversity_metrics['total_garments']}
🎨 Colors: {self.diversity_metrics['unique_colors']}
⚡ Essential: {self.diversity_metrics['essential_coverage']:.0f}%
✨ Accent: {self.diversity_metrics['accent_coverage']:.0f}%
⚖️ Balance: {self.diversity_metrics['color_balance']:.0f}%
"""

        if health_score >= 80:
            rec_text = "🏆 OUTSTANDING!\n\nExceptional diversity\nand balance achieved.\nProfessional quality!"
            rec_color = '#90EE90'
        elif health_score >= 65:
            rec_text = "🌟 EXCELLENT!\n\nGreat foundation.\nMinor improvements\npossible."
            rec_color = '#87CEEB'
        elif health_score >= 50:
            rec_text = "✅ GOOD BASE\n\nSolid foundation.\nAdd more accent\ncolors for variety."
            rec_color = '#F0E68C'
        else:
            rec_text = "📈 IMPROVE\n\nSignificant growth\nopportunity available.\nFocus on essentials."
            rec_color = '#FFA07A'

        ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))

        ax3.text(0.05, 0.42, rec_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor=rec_color, alpha=0.9))

        # Coverage analysis bar chart
        ax4 = fig.add_subplot(gs[1, :])

        categories = ['Essential\nColors', 'Accent\nColors', 'All\nColors', 'Health\nScore']
        percentages = [
            self.diversity_metrics['essential_coverage'],
            self.diversity_metrics['accent_coverage'], 
            (self.diversity_metrics['unique_colors'] / len(self.color_categories)) * 100,
            self.diversity_metrics['health_score']
        ]

        bar_colors = ['#FF6B6B', '#FFD93D', '#6BCF7F', '#4ECDC4']

        bars = ax4.bar(categories, percentages, 
                      color=bar_colors,
                      alpha=0.8, 
                      edgecolor='black', 
                      linewidth=1)

        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', 
                    fontweight='bold', fontsize=11)

        ax4.set_ylabel('Score / Coverage (%)', fontweight='bold', fontsize=11)
        ax4.set_title('📊 Coverage & Performance Analysis', 
                     fontsize=13, fontweight='bold', pad=15)
        ax4.set_ylim(0, 105)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_axisbelow(True)

        # Technical details
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        tech_details = f"""🔬 TECHNICAL DETAILS: Image: {self.analysis_stats['image_dimensions']} | Regions: {self.analysis_stats['total_regions_detected']} | Success: {self.analysis_stats['successful_color_analyses']} | Avg Colors: {self.analysis_stats['average_colors_per_garment']:.1f} | Top Color: {self.analysis_stats['most_common_color'].title()}

🛠️  METHODS: Multi-region detection • RGB-HSV classification • Shannon entropy diversity • K-means clustering • Confidence scoring"""

        ax5.text(0.02, 0.8, tech_details, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='#F0F8FF', alpha=0.8))

        # Title and timestamp
        fig.suptitle('🎨 Professional Wardrobe Color Analysis Report (FIXED v2.1) 🎨', 
                    fontsize=16, fontweight='bold', y=0.96)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.02, 0.01, f"Generated: {timestamp} | Patent-Quality Analysis | Fixed Display Version", 
                fontsize=8, alpha=0.7)

        plt.tight_layout()

        # FIXED save method - remove the problematic 'quality' parameter
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"✅ Visualization saved: {save_path}")
        except Exception as e:
            print(f"⚠️  Save warning: {e}")
            # Fallback save
            plt.savefig(save_path, bbox_inches='tight', facecolor='white')
            print(f"✅ Visualization saved with fallback method: {save_path}")

        # FIXED display method that keeps window open
        print("🖼️  Displaying visualization...")
        plt.show(block=False)  # Non-blocking show

        # Keep the window open
        print("\n" + "="*50)
        print("🎯 VISUALIZATION DISPLAYED!")
        print("👀 The analysis window should now be visible")
        print("🔄 Close the window manually when you're done viewing")
        print("⏳ Press Enter here to continue or close the window...")
        print("="*50)

        # Wait for user input to keep program running
        try:
            input()  # Wait for user to press Enter
        except:
            time.sleep(30)  # Fallback: wait 30 seconds

        return fig

def main():
    """FIXED main execution function"""
    print("🌟 FIXED PROFESSIONAL WARDROBE ANALYZER v2.1 🌟")
    print("Patent-Quality Implementation with FIXED Display")
    print("="*70)

    # Initialize analyzer
    analyzer = FixedProfessionalWardrobeAnalyzer()

    # IMPORTANT: UPDATE THIS PATH TO YOUR IMAGE
    image_path = r"C:\Users\nadai\Downloads\wardrob_3pic.jpg"

    try:
        print("\n🚀 Starting comprehensive analysis...")

        if analyzer.analyze_wardrobe(image_path):
            print("\n🎨 Creating FIXED visualization...")

            fig = analyzer.create_fixed_visualization(
                image_path, 
                "fixed_professional_analysis.png"
            )

            print("\n📋 Generating report...")

            # Generate basic report data
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'diversity_score': analyzer.diversity_metrics['diversity_score'],
                'health_score': analyzer.diversity_metrics['health_score'],
                'total_garments': analyzer.diversity_metrics['total_garments'],
                'unique_colors': analyzer.diversity_metrics['unique_colors'],
                'color_distribution': dict(analyzer.color_analysis),
                'garments': [
                    {
                        'id': g['id'],
                        'primary_color': g['primary_color'],
                        'bbox': g['bbox'],
                        'colors': [c['name'] for c in g['colors']]
                    }
                    for g in analyzer.garment_detections
                ]
            }

            with open('fixed_analysis_report.json', 'w') as f:
                json.dump(convert_numpy(report_data), f, indent=2)
            print("✅ Report saved: fixed_analysis_report.json")

            print("\n" + "="*70)
            print("🎊 FIXED ANALYSIS COMPLETED SUCCESSFULLY!")
            print("\n📁 Files Generated:")
            print("   1. 🖼️  fixed_professional_analysis.png")
            print("   2. 📄 fixed_analysis_report.json")
            print("\n🏆 Ready for professor presentation!")
            print("✨ Visualization should be displayed and stay open!")

        else:
            print("❌ Analysis failed. Please check the image file path.")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTROUBLESHOoting:")
        print("1. Check if the image path is correct")
        print("2. Ensure the image file exists")
        print("3. Try with a different image format")

if __name__ == "__main__":
    main()
