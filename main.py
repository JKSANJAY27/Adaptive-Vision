import os
import cv2
from garment_analyzer import analyze_garment_colors

def main():
    """Main interface for hybrid garment color analysis"""
    print("🚀 ADVANCED GARMENT COLOR ANALYZER (SAM + Enhanced Processing)")
    print("="*60)

    # Check if running in Google Colab
    try:
        from google.colab import files
        print("📱 Running in Google Colab")
        print("📤 Please upload your garment image:")
        uploaded = files.upload()
        image_path = list(uploaded.keys())[0]
        print(f"✅ Uploaded: {image_path}")

    except ImportError:
        print("💻 Running locally")
        image_path = input("📁 Enter image path: ").strip().strip('"').strip("'")

        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return

    print(f"\n🔬 Analyzing with advanced AI: {image_path}")
    print("⏳ This may take a moment for SAM model loading...")

    # Run the hybrid analysis
    result = analyze_garment_colors(image_path)

    if result[0] is not None:
        print("✅ Advanced analysis completed successfully!")

        # Unpack results
        masked_image, primary_colors, secondary_colors = result

        # Option to save the masked image
        save = input("\n💾 Save SAM-segmented image? (y/n): ").strip().lower()
        if save == 'y':
            output_path = image_path.rsplit('.', 1)[0] + '_sam_masked.jpg'
            cv2.imwrite(output_path, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved: {output_path}")

        # Option to save analysis results
        if primary_colors or secondary_colors:
            save_results = input("📄 Save advanced analysis results to text file? (y/n): ").strip().lower()
            if save_results == 'y':
                results_path = image_path.rsplit('.', 1)[0] + '_advanced_analysis.txt'
                with open(results_path, 'w') as f:
                    f.write("ADVANCED GARMENT COLOR ANALYSIS RESULTS\n")
                    f.write("(Using SAM + MeanShift/KMeans + Delta E Color Science)\n")
                    f.write("=" * 60 + "\n\n")

                    f.write("PRIMARY COLORS:\n")
                    for i, color in enumerate(primary_colors, 1):
                        f.write(f"{i}. {color['name']} {color['hex']} RGB{color['rgb']} {color['percentage']}%\n")

                    f.write("\nSECONDARY COLORS:\n")
                    for i, color in enumerate(secondary_colors, 1):
                        f.write(f"{i}. {color['name']} {color['hex']} RGB{color['rgb']} {color['percentage']}%\n")

                    f.write("\nMODELS USED:\n")
                    f.write("- SAM (Segment Anything Model) for garment segmentation\n")
                    f.write("- MeanShift clustering for chromatic color extraction\n") 
                    f.write("- K-Means clustering (fallback)\n")
                    f.write("- Delta E CIE2000 for perceptual color merging\n")
                    f.write("- LAB color space analysis\n")
                    f.write("- GrabCut segmentation (fallback)\n")

                print(f"📄 Results saved: {results_path}")
    else:
        print("❌ Advanced analysis failed - check if all dependencies are installed")

if __name__ == "__main__":
    main()