
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
from collections import Counter, defaultdict
import json
from datetime import datetime
import colorsys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import OpenCV, fallback if not available
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️  OpenCV not available, using PIL-based alternatives")

# Try to import sklearn
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn not available, using simple color extraction")

class WardrobeAnalyzer:
    def __init__(self):
        """Initialize the wardrobe analyzer"""
        # Color categories for wardrobe analysis
        self.color_categories = {
            'red': [(255, 0, 0), (220, 20, 60), (178, 34, 34), (255, 69, 0), (139, 0, 0),
                   (205, 92, 92), (255, 99, 71), (255, 160, 122)],
            'pink': [(255, 192, 203), (255, 20, 147), (255, 105, 180), (219, 112, 147),
                    (255, 182, 193), (255, 218, 185)],
            'blue': [(0, 0, 255), (0, 100, 255), (30, 144, 255), (70, 130, 180),
                    (100, 149, 237), (65, 105, 225), (135, 206, 235)],
            'lightblue': [(173, 216, 230), (135, 206, 235), (176, 196, 222), (230, 230, 250)],
            'green': [(0, 255, 0), (34, 139, 34), (0, 128, 0), (50, 205, 50),
                     (124, 252, 0), (46, 139, 87), (144, 238, 144)],
            'yellow': [(255, 255, 0), (255, 215, 0), (255, 255, 224), (255, 250, 205),
                      (240, 230, 140), (255, 228, 181)],
            'orange': [(255, 165, 0), (255, 140, 0), (255, 99, 71), (255, 127, 80),
                      (255, 160, 122), (255, 218, 185)],
            'purple': [(128, 0, 128), (147, 112, 219), (138, 43, 226), (75, 0, 130),
                      (186, 85, 211), (221, 160, 221)],
            'brown': [(165, 42, 42), (139, 69, 19), (160, 82, 45), (205, 133, 63),
                     (210, 180, 140), (222, 184, 135)],
            'black': [(0, 0, 0), (25, 25, 25), (47, 79, 79), (36, 36, 36),
                     (64, 64, 64), (105, 105, 105)],
            'white': [(255, 255, 255), (248, 248, 255), (245, 245, 245), (250, 250, 250),
                     (255, 250, 240), (240, 248, 255)],
            'gray': [(128, 128, 128), (169, 169, 169), (105, 105, 105), (192, 192, 192),
                    (211, 211, 211), (220, 220, 220)],
            'navy': [(0, 0, 128), (25, 25, 112), (72, 61, 139), (60, 60, 120)],
            'beige': [(245, 245, 220), (255, 228, 196), (222, 184, 135), (210, 180, 140)],
            'cream': [(255, 253, 208), (255, 248, 220), (253, 245, 230)],
            'maroon': [(128, 0, 0), (139, 0, 0), (165, 42, 42)]
        }

        # Define display colors for pie chart (matching the category names)
        self.display_colors = {
            'red': '#FF0000',
            'pink': '#FFC0CB', 
            'blue': '#0000FF',
            'lightblue': '#ADD8E6',
            'green': '#008000',
            'yellow': '#FFFF00',
            'orange': '#FFA500',
            'purple': '#800080',
            'brown': '#A0522D',
            'black': '#000000',
            'white': '#FFFFFF',
            'gray': '#808080',
            'navy': '#000080',
            'beige': '#F5F5DC',
            'cream': '#FFFDD0',
            'maroon': '#800000'
        }

        self.essential_colors = ['black', 'white', 'navy', 'blue', 'gray', 'red', 'brown']
        self.accent_colors = ['pink', 'green', 'yellow', 'orange', 'purple']

        # Initialize tracking variables
        self.garment_color_frequency = defaultdict(float)
        self.detected_colors = set()
        self.individual_garments = []
        self.analysis_results = {}
        self.original_image = None

    def load_image(self, image_path):
        """Load and preprocess image"""
        try:
            if isinstance(image_path, str):
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                self.original_image = Image.open(image_path)
            else:
                self.original_image = Image.open(image_path)

            if self.original_image.mode != 'RGB':
                self.original_image = self.original_image.convert('RGB')

            # Resize if too large
            max_size = 1000
            if max(self.original_image.size) > max_size:
                self.original_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            print(f"✅ Image loaded successfully: {self.original_image.size}")
            return True

        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False

    def detect_garments(self):
        """Detect garments in the wardrobe image"""
        if not self.original_image:
            return False

        image_array = np.array(self.original_image)
        height, width = image_array.shape[:2]

        print("🔍 Detecting garments...")

        # Create a grid to detect garments
        cols = 15  # Number of potential garment columns
        rows = 1   # Single row for hanging clothes

        col_width = width // cols
        garment_height = int(height * 0.8)  # 80% of image height
        start_y = int(height * 0.1)  # Start 10% from top

        garments_found = 0

        for col in range(cols):
            x = col * col_width

            # Extract column region
            if x + col_width < width:
                region = image_array[start_y:start_y + garment_height, x:x + col_width]

                if self._is_valid_garment_region(region):
                    # Create garment data
                    pixels = region.reshape(-1, 3)

                    garment_data = {
                        'id': garments_found,
                        'bbox': (x, start_y, col_width, garment_height),
                        'pixels': pixels,
                        'area': len(pixels),
                        'method': 'grid',
                        'confidence': 0.8
                    }

                    self.individual_garments.append(garment_data)
                    garments_found += 1

        print(f"✅ Detected {len(self.individual_garments)} garments")
        return len(self.individual_garments) > 0

    def _is_valid_garment_region(self, region):
        """Check if a region likely contains a garment"""
        if region.size == 0 or len(region.shape) != 3:
            return False

        # Check color variance
        std_r = np.std(region[:, :, 0])
        std_g = np.std(region[:, :, 1])
        std_b = np.std(region[:, :, 2])

        avg_std = (std_r + std_g + std_b) / 3

        # Should have some variation but not too much
        return 8 < avg_std < 100

    def extract_colors(self):
        """Extract colors from detected garments"""
        if not self.individual_garments:
            print("❌ No garments detected")
            return {}

        all_colors = {}
        print(f"🎨 Extracting colors from {len(self.individual_garments)} garments...")

        for garment in self.individual_garments:
            pixels = garment['pixels']

            if len(pixels) < 50:
                continue

            # Sample pixels for faster processing
            sample_size = min(1000, len(pixels))
            sampled_pixels = pixels[np.random.choice(len(pixels), sample_size, replace=False)]

            # Extract dominant colors
            if HAS_SKLEARN:
                colors = self._extract_colors_kmeans(sampled_pixels)
            else:
                colors = self._extract_colors_simple(sampled_pixels)

            # Add to results
            for i, color_data in enumerate(colors):
                key = f"G{garment['id']}_C{i}"
                all_colors[key] = color_data

                # Update frequency tracking
                category = color_data['category']
                self.garment_color_frequency[category] += color_data['frequency']
                self.detected_colors.add(category)

        print(f"✅ Extracted {len(all_colors)} colors")
        return all_colors

    def _extract_colors_kmeans(self, pixels):
        """Extract colors using KMeans clustering"""
        try:
            # Use 3 clusters for dominant colors
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(pixels)
            centers = kmeans.cluster_centers_

            # Calculate frequencies
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_pixels = len(labels)

            colors = []
            for center, count in zip(centers, counts):
                frequency = count / total_pixels
                if frequency > 0.1:  # At least 10% of garment
                    center = np.clip(center, 0, 255).astype(int)
                    category = self.classify_color(center)

                    if category != 'unknown':
                        colors.append({
                            'rgb': tuple(center),
                            'hex': self.rgb_to_hex(center),
                            'category': category,
                            'frequency': frequency
                        })

            return colors

        except Exception as e:
            print(f"⚠️ KMeans error: {e}")
            return self._extract_colors_simple(pixels)

    def _extract_colors_simple(self, pixels):
        """Simple color extraction without clustering"""
        # Calculate mean color
        mean_color = np.mean(pixels, axis=0).astype(int)
        category = self.classify_color(mean_color)

        if category != 'unknown':
            return [{
                'rgb': tuple(mean_color),
                'hex': self.rgb_to_hex(mean_color),
                'category': category,
                'frequency': 1.0
            }]
        return []

    def classify_color(self, rgb_color):
        """Classify RGB color into categories"""
        min_distance = float('inf')
        closest_category = 'unknown'

        rgb_color = np.clip(rgb_color, 0, 255).astype(int)

        for category, color_variants in self.color_categories.items():
            for variant_rgb in color_variants:
                distance = np.sqrt(np.sum((rgb_color - np.array(variant_rgb)) ** 2))

                if distance < min_distance:
                    min_distance = distance
                    closest_category = category

        # Dynamic threshold
        r, g, b = rgb_color
        color_range = max(r, g, b) - min(r, g, b)
        avg_brightness = np.mean([r, g, b])

        if color_range < 20:  # Neutral colors
            threshold = 45 if avg_brightness < 40 or avg_brightness > 200 else 50
        else:
            threshold = 65

        return closest_category if min_distance < threshold else 'unknown'

    def analyze_wardrobe(self):
        """Analyze wardrobe completeness"""
        if not self.detected_colors:
            return {
                'current_colors': [],
                'color_distribution': {},
                'missing_essential': self.essential_colors,
                'diversity_score': 0,
                'garment_count': 0,
                'recommendations': []
            }

        # Color distribution
        total_freq = sum(self.garment_color_frequency.values())
        color_percentages = {}
        if total_freq > 0:
            color_percentages = {
                color: (freq/total_freq)*100
                for color, freq in self.garment_color_frequency.items()
            }

        # Missing essential colors
        missing_essential = [c for c in self.essential_colors if c not in self.detected_colors]

        # Calculate diversity score
        diversity_score = self._calculate_diversity_score()

        # Generate recommendations
        recommendations = []
        if missing_essential:
            recommendations.append({
                'priority': 'HIGH',
                'type': 'Essential Colors',
                'colors': missing_essential[:3],
                'reason': 'Foundation colors for wardrobe versatility'
            })

        results = {
            'current_colors': sorted(list(self.detected_colors)),
            'color_distribution': color_percentages,
            'missing_essential': missing_essential,
            'diversity_score': diversity_score,
            'garment_count': len(self.individual_garments),
            'recommendations': recommendations
        }

        self.analysis_results = results
        return results

    def _calculate_diversity_score(self):
        """Calculate diversity score"""
        if not self.detected_colors:
            return 0

        # Variety score (50 points)
        variety_score = (len(self.detected_colors) / len(self.color_categories)) * 50

        # Essential coverage (30 points)
        essential_present = len([c for c in self.essential_colors if c in self.detected_colors])
        essential_score = (essential_present / len(self.essential_colors)) * 30

        # Garment count (20 points)
        garment_score = min(20, len(self.individual_garments) * 2)

        return min(100, variety_score + essential_score + garment_score)

    def rgb_to_hex(self, rgb):
        """Convert RGB to hex"""
        rgb = np.clip(rgb, 0, 255).astype(int)
        return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

    def generate_report(self):
        """Generate analysis report"""
        if not self.analysis_results:
            print("❌ No analysis results available")
            return

        print("\n" + "="*60)
        print(" "*20 + "🎨 WARDROBE ANALYSIS 🎨")
        print("="*60)
        print(f"📅 Analysis Date: {datetime.now().strftime('%B %d, %Y')}")
        print(f"👔 Garments Analyzed: {len(self.individual_garments)}")
        print("-"*60)

        # Diversity score
        score = self.analysis_results['diversity_score']
        if score >= 80:
            status = "EXCELLENT"
        elif score >= 60:
            status = "GOOD"
        elif score >= 40:
            status = "FAIR"
        else:
            status = "NEEDS IMPROVEMENT"

        print(f"\n🏆 DIVERSITY SCORE: {score:.1f}/100 ({status})")

        # Colors detected
        colors = self.analysis_results['current_colors']
        print(f"\n🎨 DETECTED COLORS ({len(colors)}):")
        for color in colors:
            print(f"  ✓ {color.title()}")

        # Color distribution
        dist = self.analysis_results['color_distribution']
        if dist:
            print(f"\n📊 COLOR DISTRIBUTION:")
            for color, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                print(f"  {color.title():<12} {pct:5.1f}%")

        # Missing essentials
        missing = self.analysis_results['missing_essential']
        if missing:
            print(f"\n❌ MISSING ESSENTIALS:")
            for color in missing:
                print(f"  • {color.title()}")
        else:
            print(f"\n✅ ALL ESSENTIAL COLORS PRESENT!")

        print("\n" + "="*60)

    def create_visualization(self):
        """Create visualization of the analysis with MATCHING COLORS in pie chart"""
        if not self.individual_garments:
            return

        try:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Wardrobe Color Analysis Results', fontsize=16, fontweight='bold')

            # 1. Original image with garment regions
            image_array = np.array(self.original_image)
            ax1.imshow(image_array)

            # Draw rectangles for garments
            for i, garment in enumerate(self.individual_garments):
                x, y, w, h = garment['bbox']
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
                ax1.add_patch(rect)
                ax1.text(x + w//2, y + h//2, f'Item {i+1}', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            ax1.set_title(f'Detected Garments ({len(self.individual_garments)})')
            ax1.axis('off')

            # 2. Color distribution pie chart with MATCHING COLORS
            if self.analysis_results.get('color_distribution'):
                colors = list(self.analysis_results['color_distribution'].keys())
                sizes = list(self.analysis_results['color_distribution'].values())

                # Create matching colors for pie chart
                pie_colors = []
                for color_name in colors:
                    if color_name in self.display_colors:
                        # Use predefined display color
                        pie_colors.append(self.display_colors[color_name])
                    else:
                        # Fallback to a default color
                        pie_colors.append('#CCCCCC')  # Light gray fallback

                # Handle white color visibility issue
                pie_colors_fixed = []
                for i, color in enumerate(pie_colors):
                    if color == '#FFFFFF':  # White
                        pie_colors_fixed.append('#F0F0F0')  # Light gray instead of pure white
                    elif color == '#000000':  # Black
                        pie_colors_fixed.append('#333333')  # Dark gray instead of pure black
                    else:
                        pie_colors_fixed.append(color)

                wedges, texts, autotexts = ax2.pie(sizes, labels=colors, autopct='%1.1f%%', 
                                                  startangle=90, colors=pie_colors_fixed)

                # Make percentage text more readable
                for autotext in autotexts:
                    autotext.set_color('black')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)

                # Add border to white/light wedges for visibility
                for i, (wedge, color) in enumerate(zip(wedges, pie_colors_fixed)):
                    if color in ['#F0F0F0', '#FFFDD0', '#F5F5DC']:  # Light colors
                        wedge.set_edgecolor('black')
                        wedge.set_linewidth(1.5)

                ax2.set_title('Color Distribution (Colors Match Detected Colors!)')
            else:
                ax2.text(0.5, 0.5, 'No color data', ha='center', va='center')
                ax2.set_title('Color Distribution')

            # 3. Metrics bar chart
            metrics = ['Diversity\nScore', 'Garments', 'Colors']
            values = [self.analysis_results['diversity_score'], 
                     len(self.individual_garments),
                     len(self.detected_colors)]

            bars = ax3.bar(metrics, values, color=['#3498db', '#2ecc71', '#e74c3c'])
            ax3.set_title('Analysis Metrics')

            # Add values on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.1f}' if isinstance(value, float) else str(value),
                        ha='center', va='bottom', fontweight='bold')

            # 4. Summary with color legend
            ax4.axis('off')
            summary_text = "ANALYSIS SUMMARY:\n\n"
            summary_text += f"• {len(self.individual_garments)} garments detected\n"
            summary_text += f"• {len(self.detected_colors)} colors identified\n"
            summary_text += f"• Diversity score: {self.analysis_results['diversity_score']:.1f}/100\n\n"

            # Add detected colors with their actual colors
            if self.detected_colors:
                summary_text += "Detected Colors:\n"
                for color in sorted(self.detected_colors):
                    summary_text += f"• {color.title()}\n"
                summary_text += "\n"

            missing = self.analysis_results.get('missing_essential', [])
            if missing:
                summary_text += f"Missing essentials:\n"
                for color in missing[:3]:
                    summary_text += f"• {color.title()}\n"
            else:
                summary_text += "Excellent color diversity!\nWell-balanced wardrobe."

            ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                    facecolor="lightblue", alpha=0.3))
            ax4.set_title('Summary & Color Legend')

            plt.tight_layout()
            plt.savefig('wardrobe_analysis.png', dpi=150, bbox_inches='tight')
            plt.show()
            print("✅ Visualization saved as 'wardrobe_analysis.png'")
            print("🎨 Pie chart colors now match the actual detected colors!")

        except Exception as e:
            print(f"⚠️ Visualization error: {e}")


def analyze_wardrobe(image_path, visualize=True):
    """
    Main function to analyze wardrobe colors and diversity
    """
    print("🚀 Starting Wardrobe Analysis...")
    print("="*50)

    try:
        analyzer = WardrobeAnalyzer()

        # Load image
        print("\n📸 Loading Image...")
        if not analyzer.load_image(image_path):
            return None

        # Detect garments
        print("\n🔍 Detecting Garments...") 
        if not analyzer.detect_garments():
            print("❌ No garments detected")
            return None

        # Extract colors
        print("\n🎨 Extracting Colors...")
        colors = analyzer.extract_colors()
        if not colors:
            print("❌ No colors extracted")
            return None

        # Analyze wardrobe
        print("\n📊 Analyzing Wardrobe...")
        results = analyzer.analyze_wardrobe()

        # Generate report
        print("\n📋 Generating Report...")
        analyzer.generate_report()

        # Create visualization
        if visualize:
            print("\n📈 Creating Visualization...")
            analyzer.create_visualization()

        print("\n🎉 ANALYSIS COMPLETE!")
        return analyzer

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🌟 WARDROBE COLOR ANALYZER 🌟")
    print("Analyze your wardrobe colors and diversity!")
    analyzer = analyze_wardrobe("idealwardrobeimage.jpg")
