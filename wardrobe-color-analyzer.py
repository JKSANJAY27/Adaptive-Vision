import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import json
import os
from skimage import segmentation, color as sk_color, filters, measure
from skimage.morphology import opening, closing
import warnings
warnings.filterwarnings('ignore')

class WardrobePhotoColorAnalyzer:
    def __init__(self):
        self.color_categories = {
            'red': [(255, 0, 0), (220, 20, 60), (178, 34, 34), (139, 0, 0), (255, 99, 71)],
            'blue': [(0, 0, 255), (0, 100, 255), (30, 144, 255), (70, 130, 180), (100, 149, 237)],
            'green': [(0, 255, 0), (34, 139, 34), (0, 128, 0), (85, 107, 47), (124, 252, 0)],
            'yellow': [(255, 255, 0), (255, 215, 0), (255, 165, 0), (218, 165, 32), (255, 255, 224)],
            'purple': [(128, 0, 128), (147, 112, 219), (138, 43, 226), (75, 0, 130), (186, 85, 211)],
            'orange': [(255, 165, 0), (255, 140, 0), (255, 69, 0), (255, 99, 71), (255, 160, 122)],
            'pink': [(255, 192, 203), (255, 20, 147), (199, 21, 133), (219, 112, 147), (255, 182, 193)],
            'brown': [(165, 42, 42), (139, 69, 19), (160, 82, 45), (210, 180, 140), (205, 133, 63)],
            'black': [(0, 0, 0), (25, 25, 25), (47, 79, 79), (105, 105, 105), (64, 64, 64)],
            'white': [(255, 255, 255), (248, 248, 255), (245, 245, 220), (230, 230, 250), (255, 250, 250)],
            'gray': [(128, 128, 128), (169, 169, 169), (192, 192, 192), (105, 105, 105), (176, 196, 222)],
            'navy': [(0, 0, 128), (25, 25, 112), (72, 61, 139), (106, 90, 205), (123, 104, 238)]
        }
        self.wardrobe_background_colors = {
            'wood_tones': [(139, 69, 19), (160, 82, 45), (210, 180, 140), (222, 184, 135), (245, 245, 220)],
            'metal_tones': [(192, 192, 192), (169, 169, 169), (128, 128, 128), (105, 105, 105)],
            'white_wardrobe': [(255, 255, 255), (248, 248, 255), (240, 240, 240), (250, 250, 250)]
        }
        self.essential_colors = ['black', 'white', 'navy', 'gray', 'red', 'blue', 'brown']
        self.seasonal_colors = {
            'spring': ['pink', 'yellow', 'green', 'white'],
            'summer': ['blue', 'white', 'yellow', 'pink'],
            'autumn': ['orange', 'brown', 'red', 'yellow'],
            'winter': ['black', 'navy', 'gray', 'purple']
        }
        self.garment_color_frequency = defaultdict(float)
        self.detected_colors = set()
        self.clothing_regions = []
        self.analysis_results = {}
        self.original_image = None

    def load_and_preprocess_image(self, image_path):
        try:
            self.original_image = Image.open(image_path)
            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')
            max_size = 1200
            width, height = self.original_image.size
            if width > max_size or height > max_size:
                scale = min(max_size/width, max_size/height)
                new_width, new_height = int(width * scale), int(height * scale)
                self.original_image = self.original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return True
        except Exception:
            return False

    def detect_clothing_regions(self, image_path, debug=False):
        if not self.load_and_preprocess_image(image_path):
            return False
        try:
            image_array = np.array(self.original_image)
            clothing_mask = self._segment_clothing_by_color(image_array)
            edge_mask = self._segment_by_edges(image_array)
            superpixel_mask = self._segment_by_superpixels(image_array)
            combined_mask = np.logical_or(np.logical_or(clothing_mask, edge_mask), superpixel_mask)
            cleaned_mask = self._clean_segmentation_mask(combined_mask)
            self._extract_clothing_regions(image_array, cleaned_mask)
            if debug:
                self._visualize_segmentation(image_array, clothing_mask, edge_mask, superpixel_mask, cleaned_mask)
            return len(self.clothing_regions) > 0
        except Exception:
            return False

    def _segment_clothing_by_color(self, image_array):
        height, width, _ = image_array.shape
        mask = np.ones((height, width), dtype=bool)
        hsv_image = sk_color.rgb2hsv(image_array)
        brightness = np.mean(image_array, axis=2)
        mask[brightness > 240] = False
        mask[brightness < 15] = False
        saturation = hsv_image[:, :, 1]
        mask[saturation < 0.1] = False
        for bg_colors in self.wardrobe_background_colors.values():
            for bg_rgb in bg_colors:
                color_diff = np.sqrt(np.sum((image_array - np.array(bg_rgb))**2, axis=2))
                mask[color_diff < 50] = False
        return mask

    def _segment_by_edges(self, image_array):
        gray = sk_color.rgb2gray(image_array)
        edges = filters.sobel(gray)
        edge_mask = edges > 0.1
        edge_mask = closing(opening(edge_mask))
        return edge_mask

    def _segment_by_superpixels(self, image_array):
        segments = segmentation.slic(image_array, n_segments=100, compactness=10, start_label=1, channel_axis=2)
        mask = np.zeros(segments.shape, dtype=bool)
        for segment_id in np.unique(segments):
            segment_mask = segments == segment_id
            segment_pixels = image_array[segment_mask]
            if len(segment_pixels) > 10:
                avg_color = np.mean(segment_pixels, axis=0)
                if not self._is_background_color_array(avg_color):
                    mask[segment_mask] = True
        return mask

    def _clean_segmentation_mask(self, mask):
        binary_mask = mask.astype(np.uint8)
        cleaned = opening(binary_mask, footprint=np.ones((5, 5)))
        cleaned = closing(cleaned, footprint=np.ones((7, 7)))
        return cleaned.astype(bool)

    def _extract_clothing_regions(self, image_array, mask):
        labeled_mask = measure.label(mask)
        self.clothing_regions = []
        min_area = mask.size * 0.001
        for region in measure.regionprops(labeled_mask):
            if region.area > min_area:
                region_mask = labeled_mask == region.label
                clothing_pixels = image_array[region_mask]
                min_row, min_col, max_row, max_col = region.bbox
                self.clothing_regions.append({
                    'pixels': clothing_pixels,
                    'mask': region_mask,
                    'bbox': (min_col, min_row, max_col - min_col, max_row - min_row),
                    'area': region.area,
                    'centroid': region.centroid
                })

    def extract_colors_from_clothing(self, k_clusters=8):
        if not self.clothing_regions:
            return {}
        all_clothing_pixels = []
        for region_data in self.clothing_regions:
            pixels = region_data['pixels']
            brightness = np.mean(pixels, axis=1)
            filtered_pixels = pixels[(brightness > 20) & (brightness < 240)]
            if len(filtered_pixels) > 30:
                all_clothing_pixels.extend(filtered_pixels.tolist())
        if not all_clothing_pixels:
            return {}
        pixels_array = np.array(all_clothing_pixels, dtype=np.float32)
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        kmeans.fit(pixels_array)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)
        clothing_colors = {}
        for cluster_id in range(k_clusters):
            color_rgb = tuple(centers[cluster_id].astype(int))
            frequency = label_counts[cluster_id] / total_pixels
            if not self._is_background_color_array(color_rgb) and frequency > 0.03:
                category = self.classify_color_category(color_rgb)
                if category != 'unknown':
                    clothing_colors[f"{category}_{cluster_id}"] = {
                        'rgb': color_rgb,
                        'hex': self.rgb_to_hex(color_rgb),
                        'category': category,
                        'frequency': frequency
                    }
                    self.garment_color_frequency[category] += frequency
                    self.detected_colors.add(category)
        return clothing_colors

    def _is_background_color_array(self, rgb_color, threshold=60):
        r, g, b = rgb_color
        for color_list in self.wardrobe_background_colors.values():
            for bg_color in color_list:
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb_color, bg_color)))
                if distance < threshold:
                    return True
        if r > 235 and g > 235 and b > 235:
            return True
        if r < 25 and g < 25 and b < 25:
            return True
        if 60 < r < 180 and 40 < g < 140 and 20 < b < 100:
            saturation = (max(r, g, b) - min(r, g, b)) / max(r, g, b) if max(r, g, b) > 0 else 0
            if saturation < 0.3:
                return True
        return False

    def classify_color_category(self, rgb_color):
        min_distance = float('inf')
        closest_category = 'unknown'
        for category, color_variants in self.color_categories.items():
            for variant_rgb in color_variants:
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb_color, variant_rgb)))
                if distance < min_distance:
                    min_distance = distance
                    closest_category = category
        if min_distance < 100:
            return closest_category
        return 'unknown'

    def analyze_wardrobe_completeness(self):
        if not self.detected_colors:
            return {}
        total_frequency = sum(self.garment_color_frequency.values())
        color_percentages = {color: (freq/total_frequency)*100 for color, freq in self.garment_color_frequency.items()} if total_frequency > 0 else {}
        missing_essential = [color for color in self.essential_colors if color not in self.detected_colors]
        underrepresented = [color for color, percentage in color_percentages.items() if percentage < 8.0]
        all_categories = set(self.color_categories.keys())
        completely_missing = all_categories - self.detected_colors
        diversity_score = self._calculate_diversity_score()
        recommendations = self._generate_recommendations(missing_essential, list(completely_missing))
        results = {
            'current_colors': list(self.detected_colors),
            'color_distribution': color_percentages,
            'missing_essential': missing_essential,
            'underrepresented': underrepresented,
            'completely_missing': list(completely_missing),
            'diversity_score': diversity_score,
            'recommendations': recommendations
        }
        self.analysis_results = results
        return results

    def _calculate_diversity_score(self):
        variety_score = (len(self.detected_colors) / len(self.color_categories)) * 50
        essential_present = len([c for c in self.essential_colors if c in self.detected_colors])
        essential_bonus = (essential_present / len(self.essential_colors)) * 30
        if self.garment_color_frequency:
            frequencies = list(self.garment_color_frequency.values())
            total = sum(frequencies)
            if total > 0:
                normalized_freq = [f/total for f in frequencies]
                entropy = -sum(p * np.log2(p) for p in normalized_freq if p > 0)
                max_entropy = np.log2(len(frequencies)) if len(frequencies) > 1 else 1
                balance_score = (entropy / max_entropy) * 20
            else:
                balance_score = 0
        else:
            balance_score = 0
        return min(100, variety_score + essential_bonus + balance_score)

    def _generate_recommendations(self, missing_essential, completely_missing):
        recommendations = []
        if missing_essential:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'Essential Colors',
                'colors': missing_essential[:3],
                'reason': 'Foundation colors that work with everything',
                'garments': [self._get_garment_suggestions(color) for color in missing_essential[:3]]
            })
        if completely_missing:
            accent_colors = [c for c in completely_missing if c in ['red', 'green', 'purple', 'yellow']]
            if accent_colors:
                recommendations.append({
                    'priority': 'MEDIUM',
                    'type': 'Accent Colors',
                    'colors': accent_colors[:2],
                    'reason': 'Add personality and styling options',
                    'garments': [self._get_garment_suggestions(color) for color in accent_colors[:2]]
                })
        return recommendations

    def _get_garment_suggestions(self, color):
        garment_map = {
            'black': {'items': ['Basic T-shirt', 'Dress pants', 'Blazer'], 'priority': 'Essential'},
            'white': {'items': ['Button-down shirt', 'White sneakers', 'Basic tee'], 'priority': 'Essential'},
            'navy': {'items': ['Navy blazer', 'Chinos', 'Sweater'], 'priority': 'Essential'},
            'gray': {'items': ['Hoodie', 'Sweatpants', 'Cardigan'], 'priority': 'Essential'},
            'red': {'items': ['Red blouse', 'Accessories', 'Statement piece'], 'priority': 'Accent'},
            'blue': {'items': ['Denim jacket', 'Blue shirt', 'Casual wear'], 'priority': 'Versatile'},
            'green': {'items': ['Green sweater', 'Olive jacket', 'Casual top'], 'priority': 'Seasonal'},
            'brown': {'items': ['Brown leather jacket', 'Boots', 'Belt'], 'priority': 'Neutral'},
            'yellow': {'items': ['Yellow top', 'Summer dress', 'Accessories'], 'priority': 'Seasonal'},
            'purple': {'items': ['Purple blouse', 'Accessories', 'Statement piece'], 'priority': 'Creative'},
            'pink': {'items': ['Pink shirt', 'Accessories', 'Casual dress'], 'priority': 'Soft'},
            'orange': {'items': ['Orange sweater', 'Autumn accessories', 'Scarf'], 'priority': 'Seasonal'}
        }
        return garment_map.get(color, {'items': ['Basic garment'], 'priority': 'Optional'})

    def generate_missing_colors_rgb(self):
        if not self.analysis_results:
            return {}
        missing_colors_rgb = {}
        all_missing = (self.analysis_results.get('missing_essential', []) + 
                      self.analysis_results.get('completely_missing', []))
        for color in set(all_missing):
            if color in self.color_categories:
                missing_colors_rgb[color] = {
                    'primary_rgb': self.color_categories[color][0],
                    'hex': self.rgb_to_hex(self.color_categories[color][0]),
                    'variants': self.color_categories[color],
                    'category': color
                }
        return missing_colors_rgb

    def create_shopping_list(self, budget_level='medium'):
        if not self.analysis_results:
            return []
        budget_limits = {'low': 3, 'medium': 6, 'high': 10}
        max_items = budget_limits.get(budget_level, 6)
        shopping_list = []
        priority_colors = []
        for color in self.analysis_results.get('missing_essential', []):
            priority_colors.append((color, 'HIGH', 'Essential'))
        useful_missing = ['red', 'green', 'blue', 'purple', 'brown']
        for color in self.analysis_results.get('completely_missing', []):
            if color in useful_missing and color not in [item[0] for item in priority_colors]:
                priority_colors.append((color, 'MEDIUM', 'Enhancement'))
        for i, (color, priority, reason) in enumerate(priority_colors[:max_items]):
            if color in self.color_categories:
                rgb = self.color_categories[color][0]
                hex_code = self.rgb_to_hex(rgb)
                garment_info = self._get_garment_suggestions(color)
                shopping_list.append({
                    'rank': i + 1,
                    'color': color.capitalize(),
                    'priority': priority,
                    'reason': reason,
                    'rgb': rgb,
                    'hex': hex_code,
                    'suggested_items': garment_info['items'][:2],
                    'styling_tip': self._get_styling_tip(color)
                })
        return shopping_list

    def _get_styling_tip(self, color):
        tips = {
            'black': 'Versatile base - pairs with everything, essential for formal wear',
            'white': 'Clean foundation - brightens outfits, perfect for layering',
            'navy': 'Professional alternative to black - works for business and casual',
            'gray': 'Perfect neutral - complements both warm and cool colors',
            'red': 'Bold statement color - use as accent, boosts confidence',
            'blue': 'Universally flattering - calming and trustworthy appearance',
            'green': 'Nature-inspired - great for casual wear and earth tones',
            'brown': 'Warm, grounding color - excellent for autumn and leather goods',
            'yellow': 'Energetic and cheerful - best in small doses or summer',
            'purple': 'Creative and unique - perfect for making statements',
            'pink': 'Soft and versatile - ranges from subtle to bold',
            'orange': 'Warm and vibrant - ideal for autumn and energetic looks'
        }
        return tips.get(color, 'Great addition to expand your color palette!')

    def rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def create_visualizations(self):
        if not self.analysis_results:
            return
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Wardrobe Color Analysis Results', fontsize=16, fontweight='bold')
        distribution = self.analysis_results.get('color_distribution', {})
        if distribution:
            colors_list = list(distribution.keys())
            percentages = list(distribution.values())
            pie_colors = []
            for color in colors_list:
                if color in self.color_categories:
                    rgb = [c/255.0 for c in self.color_categories[color][0]]
                    pie_colors.append(rgb)
                else:
                    pie_colors.append([0.5, 0.5, 0.5])
            axes[0, 0].pie(percentages, labels=colors_list, colors=pie_colors, autopct='%1.1f%%', startangle=90)
            axes[0, 0].set_title('Current Color Distribution')
        diversity_score = self.analysis_results.get('diversity_score', 0)
        theta = np.linspace(0, np.pi, 100)
        axes[0, 1].plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=15, alpha=0.3)
        if diversity_score > 0:
            score_theta = np.linspace(0, np.pi * (diversity_score/100), max(1, int(diversity_score)))
            color = 'red' if diversity_score < 40 else 'orange' if diversity_score < 70 else 'green'
            axes[0, 1].plot(np.cos(score_theta), np.sin(score_theta), color=color, linewidth=15, alpha=0.8)
        axes[0, 1].text(0, -0.3, f'{diversity_score:.1f}/100', ha='center', va='center', fontsize=16, fontweight='bold')
        axes[0, 1].set_title('Diversity Score')
        axes[0, 1].set_xlim(-1.2, 1.2)
        axes[0, 1].set_ylim(-0.5, 1.2)
        axes[0, 1].axis('off')
        missing_essential = self.analysis_results.get('missing_essential', [])
        if missing_essential:
            missing_colors = []
            for color in missing_essential[:5]:
                if color in self.color_categories:
                    rgb = [c/255.0 for c in self.color_categories[color][0]]
                    missing_colors.append(rgb)
            if missing_colors:
                axes[1, 0].bar(range(len(missing_essential[:5])), [1]*len(missing_essential[:5]),
                               color=missing_colors, alpha=0.8)
                axes[1, 0].set_title('Missing Essential Colors')
                axes[1, 0].set_xticks(range(len(missing_essential[:5])))
                axes[1, 0].set_xticklabels([c.capitalize() for c in missing_essential[:5]], rotation=45)
                axes[1, 0].set_yticks([])
        total_categories = len(self.color_categories)
        present_count = len(self.detected_colors)
        missing_count = total_categories - present_count
        axes[1, 1].bar(['Present', 'Missing'], [present_count, missing_count], color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_title('Color Categories Overview')
        axes[1, 1].set_ylabel('Number of Categories')
        plt.tight_layout()
        plt.savefig('wardrobe_analysis_complete.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results(self, filepath='wardrobe_analysis_results.json'):
        missing_colors_rgb = self.generate_missing_colors_rgb()
        shopping_list = self.create_shopping_list()
        complete_results = {
            'analysis_summary': {
                'total_clothing_regions': len(self.clothing_regions),
                'colors_detected': len(self.detected_colors),
                'diversity_score': self.analysis_results.get('diversity_score', 0),
                'image_size': self.original_image.size if self.original_image else None
            },
            'detected_colors': {
                'colors_found': list(self.detected_colors),
                'color_distribution': self.analysis_results.get('color_distribution', {}),
                'detailed_frequencies': dict(self.garment_color_frequency)
            },
            'missing_analysis': {
                'missing_essential': self.analysis_results.get('missing_essential', []),
                'completely_missing': self.analysis_results.get('completely_missing', []),
                'underrepresented': self.analysis_results.get('underrepresented', [])
            },
            'recommendations': {
                'rgb_values': missing_colors_rgb,
                'shopping_list': shopping_list,
                'improvement_suggestions': self.analysis_results.get('recommendations', [])
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(complete_results, f, indent=2)
        return filepath

def create_sample_wardrobe_image():
    img_width, img_height = 800, 600
    image = Image.new('RGB', (img_width, img_height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    frame_color = (180, 180, 180)
    draw.rectangle([40, 40, img_width-40, img_height-40], outline=frame_color, width=15)
    clothes_data = [
        (80, 80, 70, 100, (0, 0, 0), 'shirt'),
        (170, 80, 70, 100, (255, 255, 255), 'shirt'),
        (260, 80, 70, 100, (0, 0, 255), 'shirt'),
        (350, 80, 70, 100, (128, 128, 128), 'shirt'),
        (440, 80, 70, 100, (255, 0, 0), 'shirt'),
        (530, 80, 70, 100, (0, 128, 0), 'shirt'),
        (620, 80, 70, 100, (128, 0, 128), 'shirt'),
        (80, 250, 70, 120, (0, 0, 128), 'pants'),
        (170, 250, 70, 120, (139, 69, 19), 'pants'),
        (260, 250, 70, 120, (0, 0, 0), 'pants'),
        (350, 250, 70, 120, (70, 130, 180), 'pants'),
        (440, 250, 70, 120, (105, 105, 105), 'pants'),
        (530, 250, 70, 120, (34, 139, 34), 'pants'),
        (80, 420, 60, 130, (255, 20, 147), 'dress'),
        (160, 420, 60, 130, (255, 140, 0), 'dress'),
        (240, 420, 60, 130, (75, 0, 130), 'dress'),
    ]
    for x, y, w, h, color, item_type in clothes_data:
        draw.rectangle([x, y, x+w, y+h], fill=color, outline=(0, 0, 0), width=1)
        if item_type == 'shirt':
            collar_color = tuple(max(0, c-30) for c in color)
            draw.rectangle([x+10, y, x+w-10, y+15], fill=collar_color)
            for btn_y in range(y+20, y+h-10, 20):
                draw.ellipse([x+w//2-2, btn_y, x+w//2+2, btn_y+4], fill=(50, 50, 50))
        elif item_type == 'pants':
            draw.line([x+w//2, y, x+w//2, y+h], fill=(0, 0, 0), width=2)
        if y < 200:
            hanger_color = (100, 100, 100)
            draw.ellipse([x+w//2-3, y-8, x+w//2+3, y-2], fill=hanger_color)
            draw.line([x+w//2, y-8, x+w//2, y-15], fill=hanger_color, width=2)
    image.save('sample_wardrobe.jpg', 'JPEG', quality=95)
    return 'sample_wardrobe.jpg'

def analyze_wardrobe_photo(image_path=None, budget_level='medium', debug=False):
    if image_path is None:
        image_path = create_sample_wardrobe_image()
    analyzer = WardrobePhotoColorAnalyzer()
    if not analyzer.detect_clothing_regions(image_path, debug=debug):
        return None
    clothing_colors = analyzer.extract_colors_from_clothing()
    if not clothing_colors:
        return None
    analysis = analyzer.analyze_wardrobe_completeness()
    analyzer.generate_missing_colors_rgb()
    analyzer.create_shopping_list(budget_level)
    analyzer.create_visualizations()
    analyzer.save_results()
    return analyzer

def quick_wardrobe_analysis(image_path, budget='medium'):
    analyzer = analyze_wardrobe_photo(image_path, budget_level=budget, debug=False)
    if analyzer:
        return analyzer
    else:
        return None

if __name__ == "__main__":
    demo_analyzer = analyze_wardrobe_photo(budget_level='medium')
