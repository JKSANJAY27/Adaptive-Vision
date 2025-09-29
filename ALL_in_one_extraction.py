# Upload this full script in a Colab cell and run after installing dependencies manually:
# !pip install mediapipe==0.10.21 opencv-python-headless scikit-learn matplotlib webcolors packaging

import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans
import webcolors
import matplotlib.pyplot as plt

def rgb_to_hex(rgb):
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

def get_color_name(rgb):
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        try:
            closest = min(webcolors.CSS3_HEX_TO_NAMES.items(),
                          key=lambda x: sum((a-b)**2 for a,b in zip(rgb, webcolors.hex_to_rgb(x[0]))))
            return closest[1].replace('_', ' ').title()
        except:
            r, g, b = rgb
            if max(rgb) - min(rgb) < 25:
                if r > 220: return "White"
                elif r > 180: return "Light Gray"
                elif r > 120: return "Medium Gray"
                elif r > 60: return "Dark Gray"
                else: return "Black"
            if r > max(g, b) + 20:
                if g > 100: return "Orange"
                elif b > 100: return "Pink"
                else: return "Red"
            elif g > max(r, b) + 20:
                if r > 100: return "Yellow"
                else: return "Green"
            elif b > max(r, g) + 20:
                if r > 100: return "Purple"
                else: return "Blue"
            else:
                if r > 150 and g > 150: return "Yellow"
                elif r > 150 and b > 150: return "Magenta"
                elif g > 150 and b > 150: return "Cyan"
                else: return "Brown"

def segment_person(rgb_image):
    mp_selfie = mp.solutions.selfie_segmentation
    with mp_selfie.SelfieSegmentation(model_selection=1) as segmenter:
        results = segmenter.process(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        mask = (results.segmentation_mask > 0.2).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)

def detect_face_bbox(rgb_image):
    mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    results = mp_face.process(cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    if results.detections:
        det = results.detections[0]
        bboxC = det.location_data.relative_bounding_box
        H, W = rgb_image.shape[:2]
        x1 = int(bboxC.xmin * W)
        y1 = int(bboxC.ymin * H)
        x2 = int((bboxC.xmin + bboxC.width) * W)
        y2 = int((bboxC.ymin + bboxC.height) * H)
        return max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    return None

def skin_mask_from_face(rgb_image, person_mask):
    face_bbox = detect_face_bbox(rgb_image)
    skin_mask = np.zeros(person_mask.shape, dtype=bool)
    if face_bbox is not None:
        x1, y1, x2, y2 = face_bbox
        y2_extra = min(person_mask.shape[0], y2 + int(0.3 * (y2 - y1)))
        x1 = max(x1 - 10, 0)
        x2 = min(x2 + 10, person_mask.shape[1])
        skin_mask[y1:y2_extra, x1:x2] = True
    skin_mask &= person_mask
    if np.sum(skin_mask) == 0:
        return skin_mask
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    ycrcb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    hsv_mask = cv2.inRange(hsv, np.array([0, 15, 50]), np.array([25, 180, 255]))
    ycrcb_mask = cv2.inRange(ycrcb, np.array([70, 130, 85]), np.array([255, 180, 135]))
    combined_mask = (hsv_mask | ycrcb_mask).astype(bool)
    skin_pixels = combined_mask & skin_mask
    num_labels, labels_im = cv2.connectedComponents(skin_pixels.astype(np.uint8))
    if num_labels > 1:
        largest_label = max(range(1, num_labels), key=lambda i: (labels_im == i).sum())
        skin_pixels = (labels_im == largest_label)
    return skin_pixels

def get_clothing_mask(person_mask, skin_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clothing_mask = person_mask & (~cv2.morphologyEx(skin_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool))
    num_labels, labels_im = cv2.connectedComponents(clothing_mask.astype(np.uint8))
    if num_labels > 1:
        largest_label = max(range(1, num_labels), key=lambda i: (labels_im == i).sum())
        clothing_mask = (labels_im == largest_label)
    return clothing_mask

def extract_skin_tone(rgb, mask):
    if not np.any(mask):
        return None, None
    skin_pixels = rgb[mask]
    median_rgb = np.median(skin_pixels, axis=0).astype(int)
    lab = cv2.cvtColor(np.uint8([[median_rgb]]), cv2.COLOR_RGB2LAB)[0, 0]
    a_star = lab[1] - 128
    b_star = lab[2] - 128
    if b_star > 10:
        undertone = "Warm"
    elif b_star < -8:
        undertone = "Cool"
    else:
        undertone = "Neutral"
    return tuple(median_rgb), undertone

def extract_garment_colors(rgb, mask):
    if not np.any(mask):
        return [], []
    garment_pixels = rgb[mask]
    n_colors = 2  # For solid garments, just 2 clusters: dominant and shadow/specular
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    kmeans.fit(garment_pixels)
    centers = kmeans.cluster_centers_.astype(int)
    labels = kmeans.predict(garment_pixels)
    unique_labels, counts = np.unique(labels, return_counts=True)
    proportions = counts / len(garment_pixels)
    sort_idx = np.argsort(proportions)[::-1]
    centers = centers[unique_labels[sort_idx]]
    proportions = proportions[sort_idx]

    # Only return one primary if the first cluster is dominant
    if proportions[0] > 0.65:
        primary = [{
            "rgb": tuple(centers[0]),
            "hex": rgb_to_hex(centers[0]),
            "name": get_color_name(centers[0]),
            "percentage": round(proportions[0]*100, 1)
        }]
        secondary = []
    else:
        # Report additional colors only if they are a major sub-region
        primary = []
        secondary = []
        for i, prop in enumerate(proportions):
            color_info = {
                "rgb": tuple(centers[i]),
                "hex": rgb_to_hex(centers[i]),
                "name": get_color_name(centers[i]),
                "percentage": round(prop*100, 1)
            }
            if prop >= 0.2:
                primary.append(color_info)
            elif prop >= 0.1:
                secondary.append(color_info)
    return primary, secondary


def apply_mask_rgb(rgb, mask):
    if mask.dtype != bool:
        mask = mask.astype(bool)
    masked = np.zeros_like(rgb)
    for c in range(3):
        masked[:, :, c] = rgb[:, :, c] * mask
    return masked

def analyze_image(image_path):
    if not os.path.isfile(image_path):
        print(f"File not found: {image_path}")
        return
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"Failed to load image: {image_path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    person_mask = segment_person(rgb)
    skin_mask = skin_mask_from_face(rgb, person_mask)
    if not np.any(skin_mask):
        print("Skin not detected; skin analysis skipped")
        skin_mask = np.zeros_like(person_mask, dtype=bool)
    clothing_mask = get_clothing_mask(person_mask, skin_mask)
    skin_rgb, undertone = extract_skin_tone(rgb, skin_mask) if np.any(skin_mask) else (None, None)
    primary_colors, secondary_colors = extract_garment_colors(rgb, clothing_mask)

    fg_rgb = apply_mask_rgb(rgb, person_mask)
    skin_rgb_img = apply_mask_rgb(rgb, skin_mask)
    clothing_rgb = apply_mask_rgb(rgb, clothing_mask)

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 4, 1)
    plt.imshow(rgb)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(fg_rgb)
    plt.title("Person Foreground")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(skin_rgb_img)
    plt.title("Skin Regions")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(clothing_rgb)
    plt.title("Garment Regions")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    print("=== Garment Colors ===")
    for c in primary_colors:
        print(f"Primary: {c['name']} | {c['hex']} | {c['percentage']}%")
    for c in secondary_colors:
        print(f"Secondary: {c['name']} | {c['hex']} | {c['percentage']}%")

    print("\n=== Skin Color ===")
    if skin_rgb is not None:
        print(f"Skin RGB: {skin_rgb}")
        print(f"Skin undertone: {undertone}")
    else:
        print("Skin region not available.")

def main():
    image_path = "white.jpg"
    analyze_image(image_path)

if __name__ == "__main__":
    main()
