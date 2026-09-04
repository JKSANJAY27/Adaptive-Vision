# Adaptive Vision
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/JKSANJAY27/Adaptive-Vision)

Adaptive Vision is a sophisticated fashion technology toolkit that uses computer vision and machine learning to analyze garments, extract color palettes, and provide personalized color recommendations. The system is designed to be highly adaptive, with a core focus on accommodating users with Color Vision Deficiencies (CVD) by simulating and optimizing for their unique perception.

The project integrates multiple cutting-edge techniques, including person and garment segmentation with Meta's Segment Anything Model (SAM) and MediaPipe, context-aware color analysis using fuzzy logic, and a multi-objective recommendation engine that considers color harmony, emotional context, and perceptual distinguishability.

## Key Features

*   **Advanced Garment Segmentation**: Utilizes multiple models for robust object isolation, including Facebook's Segment Anything Model (SAM), MediaPipe, and OpenCV's Mask R-CNN, with GrabCut as a fallback.
*   **Intelligent Color Extraction**: Employs clustering algorithms (MeanShift, K-Means) in the CIELAB color space to identify primary and secondary colors from garments and skin tones.
*   **CVD-Aware Recommendation Engine**: Simulates color perception for various CVD types (Protanopia, Deuteranopia, Tritanopia) and recommends colors that are both harmonious and perceptually distinct for the user.
*   **Fuzzy Logic System**: Analyzes color properties (hue, saturation, value) and evaluates harmony based on nuanced, context-aware fuzzy rules.
*   **Contextual & Emotional Analysis**: Considers fashion style, season, occasion, and the emotional associations of colors to provide relevant recommendations.
*   **User Perception Calibration**: Includes a web-based tool for users to calibrate the system to their unique color perception, adjusting for hue and saturation offsets.
*   **Interactive Web Application**: A Flask-based app demonstrates the full pipeline, allowing users to upload an image and receive a detailed analysis and color recommendation.

## Repository Structure

This repository contains a collection of scripts, models, and a web application that together form the Adaptive Vision system.

| File/Directory                  | Description                                                                                                                                                             |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core Application**            |                                                                                                                                                                         |
| `app.py`                        | The main Flask web application that integrates segmentation, color analysis, and recommendation.                                                                          |
| `templates/`                    | Contains HTML templates for the web interface, including the image upload page, results display, and user calibration tool.                                               |
| `user_profile.json`             | Stores the output of the user perception calibration (hue offset and saturation scale).                                                                                   |
| **Segmentation & Analysis**     |                                                                                                                                                                         |
| `garment_analyzer.py`           | An advanced script using Meta's SAM for high-quality garment segmentation and color extraction.                                                                           |
| `wardrobe-color-analyzer.py`    | Analyzes a full wardrobe image using Detectron2 and an adaptive grid to identify all garment colors.                                                                      |
| `ALL_in_one_extraction.py`      | A self-contained script using MediaPipe for person segmentation, skin/clothing separation, and color/undertone analysis.                                                  |
| `Segmentation_Extraction.ipynb` | A Jupyter Notebook demonstrating the garment segmentation process with SAM and subsequent color palette extraction.                                                     |
| **Recommendation Logic**        |                                                                                                                                                                         |
| `combined_recommendation.py`    | The most advanced recommendation engine, integrating fuzzy logic, context awareness, advanced CVD profiling, and multi-objective optimization.                              |
| `adaptive_recommender.py`       | A core recommender that generates accent colors based on color theory and a custom CVD profile.                                                                           |
| `fuzzy2.py`                     | An enhanced fashion recommender using a fuzzy logic system to classify colors by tone (neutral, dark, bright) and temperature (warm, cool) for outfit matching.          |
| **Models & Configuration**      |                                                                                                                                                                         |
| `mask_rcnn_*.pbtxt` / `*.pb`    | Model files for the Mask R-CNN object detection network used in `app.py`.                                                                                                 |
| `coco_labels.txt`               | A list of object classes corresponding to the COCO dataset used by the Mask R-CNN model.                                                                                  |
| `uploads/`                      | Default directory for storing images uploaded via the web application.                                                                                                    |

## Getting Started

### Prerequisites

*   Python 3.8+
*   pip
*   PyTorch (required for SAM and Detectron2)
*   A C++ compiler (required for installing some dependencies)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jksanjay27/Adaptive-Vision.git
    cd Adaptive-Vision
    ```

2.  **Install dependencies:**
    The project uses several libraries for computer vision and machine learning. You can install them using pip. Note that some scripts handle their own dependencies (e.g., `garment_analyzer.py`).

    ```bash
    pip install flask opencv-python-headless numpy scikit-learn matplotlib skfuzzy webcolors colormath gdown
    ```

3.  **Install PyTorch:**
    For models like SAM and Detectron2, PyTorch is required. Install it by following the official instructions for your platform: [pytorch.org](https://pytorch.org/get-started/locally/).

4.  **Install Detectron2 (Optional):**
    To enable wardrobe analysis with `wardrobe-color-analyzer.py`, install Detectron2:
    ```bash
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    ```

### Running the Web Application

The simplest way to use the system is through the Flask web application.

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```
    The application will start on `http://127.0.0.1:5000`.

2.  **Use the application:**
    *   Open your web browser and navigate to `http://127.0.0.1:5000`.
    *   **(Optional) Calibrate Perception:** Click the "Run Calibration" link to complete a short test that fine-tunes the system to your personal color perception. Your results are saved in `user_profile.json`.
    *   **Upload an Image:** Upload an image of a person wearing a garment.
    *   **View Results:** The system will process the image and display the segmented garment, the dominant color extracted, and a recommended complementary color, along with analysis from the fuzzy logic and ML models.

### Running Standalone Scripts

You can also run individual scripts for more targeted analysis.

*   **Advanced Garment Analysis (SAM):**
    ```bash
    python main.py
    ```
    This script will prompt for an image path and use the powerful SAM model for segmentation.

*   **Comprehensive Recommendation Demo:**
    ```bash
    python combined_recommendation.py
    ```
    This script demonstrates the full power of the context-aware, CVD-adaptive recommendation engine with detailed visual outputs.

## Core Components Explained

### 1. Garment and Person Segmentation

The system uses a multi-model approach to accurately isolate the garment or person from the background.

*   **Segment Anything Model (SAM):** Used in `garment_analyzer.py` and `Segmentation_Extraction.ipynb`, this model generates multiple mask candidates for an image. A scoring system evaluates these masks based on area, color saturation (chroma), and centrality to select the one most likely to be the main garment.
*   **MediaPipe:** The `ALL_in_one_extraction.py` script uses MediaPipe's Selfie Segmentation and Face Detection models to create precise masks for the person, skin, and clothing. This allows for separate analysis of skin tone and garment color.
*   **Mask R-CNN:** The Flask app (`app.py`) can use a pre-trained Mask R-CNN model to detect and segment a 'person' object from the image.
*   **GrabCut:** This is the fallback method used in `app.py` if the Mask R-CNN models are not present. It segments the foreground from the background based on a bounding box.

### 2. Color Analysis and Extraction

Once a garment is segmented, the system extracts its representative colors.

*   **Color Space:** Analysis is primarily performed in the **CIELAB** color space, which is designed to be more perceptually uniform than RGB. This means that numeric distances between colors in LAB space better align with how humans perceive their differences.
*   **Clustering:** **K-Means** and **MeanShift** clustering algorithms group the pixels of the garment into a small number of dominant color clusters.
*   **Color Merging:** To avoid an overly noisy palette, visually similar color clusters are merged using the **Delta E 2000** formula, a sophisticated metric for calculating perceptual color difference.
*   **Classification:** The final extracted colors are classified into human-readable names (e.g., "Navy", "Maroon", "Beige") and categorized as primary or secondary based on their percentage of the total area.

### 3. Adaptive Recommendation Engine

The recommendation logic is multi-layered, combining color theory, user perception, and context.

*   **Candidate Generation:** A pool of potential recommendation colors is generated based on established color harmony rules (complementary, triadic, analogous, etc.) applied to the input garment colors.
*   **CVD Simulation:** For users with a specified color vision deficiency, the system uses a transformation matrix to simulate how both the input colors and the candidate colors would appear to them.
*   **Multi-Objective Scoring:** Each candidate is scored against a weighted set of objectives:
    1.  **Fashion Harmony (Fuzzy Logic):** A fuzzy logic system scores the "fashion sense" of a color combination based on its tone (dark, bright, neutral) and temperature (warm, cool).
    2.  **CVD Distinguishability:** The candidate's perceptual distance (Delta E) and contrast ratio from the primary colors are measured in the *simulated* CVD color space. Higher scores are given to colors that remain distinct.
    3.  **Contextual Appropriateness:** The candidate is scored based on its emotional and cultural associations, tailored to the user-defined context (e.g., "Business Formal", "Casual Summer").
    4.  **Texture Compatibility:** The luminance difference is checked to ensure colors can be differentiated by texture, a common compensation strategy for individuals with CVD.
*   **Final Selection:** The candidates are ranked by their composite score, and the top results are presented to the user with detailed explanations and confidence levels.
