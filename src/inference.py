import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import glob
import os
import random

# --- 1. Math Helpers (With your updated tolerances!) ---
def get_slope_and_intercept(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf'), x1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def merge_lines(lines, slope_tolerance=0.1, intercept_tolerance=15):
    if lines is None: return []
    merged_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope, intercept = get_slope_and_intercept(x1, y1, x2, y2)
        added = False
        for group in merged_lines:
            g_slope, g_intercept = group['slope'], group['intercept']
            if (abs(slope - g_slope) < slope_tolerance or (slope == float('inf') and g_slope == float('inf'))) and \
               abs(intercept - g_intercept) < intercept_tolerance:
                group['points'].extend([(x1, y1), (x2, y2)])
                group['slope'] = (group['slope'] + slope) / 2 if slope != float('inf') else float('inf')
                group['intercept'] = (group['intercept'] + intercept) / 2
                added = True
                break
        if not added:
            merged_lines.append({'slope': slope, 'intercept': intercept, 'points': [(x1, y1), (x2, y2)]})

    final_lines = []
    for group in merged_lines:
        points = group['points']
        if group['slope'] == float('inf') or abs(group['slope']) > 1:
            points.sort(key=lambda p: p[1])
        else:
            points.sort(key=lambda p: p[0])
        final_lines.append((points[0][0], points[0][1], points[-1][0], points[-1][1]))
    return final_lines

# --- 2. Batch Processing Pipeline ---
def run_batch_inference(test_dir, model_path="best_drone_wire_model.pth", num_images=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Model on {device.upper()}...")

    # Load Model ONCE for the whole batch
    model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Find and shuffle test images
    all_test_images = glob.glob(os.path.join(test_dir, "*.jpg"))
    if not all_test_images:
        print(f"Error: No .jpg images found in {test_dir}")
        return

    # Pick N random images
    selected_images = random.sample(all_test_images, min(num_images, len(all_test_images)))
    print(f"Processing {len(selected_images)} random images...\n")

    # Set up the Matplotlib Grid (e.g., 2 rows, 3 columns for 6 images)
    cols = 3
    rows = (len(selected_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for idx, img_path in enumerate(selected_images):
        filename = os.path.basename(img_path)
        
        # Read and resize
        orig_img = cv2.imread(img_path)
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(orig_img_rgb, (640, 480))
        
        # Prepare for AI
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).to(device)

        # AI Prediction
        with torch.no_grad():
            raw_prediction = model(img_tensor)
            prob_mask = torch.sigmoid(raw_prediction).squeeze().cpu().numpy()

        # Classical CV Extraction
        binary_mask = (prob_mask > 0.25).astype(np.uint8) * 255
        kernel = np.ones((3,3), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        thinned_mask = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
        raw_lines = cv2.HoughLinesP(thinned_mask, 1, np.pi/180, threshold=30, minLineLength=100, maxLineGap=60)
        clean_lines = merge_lines(raw_lines)
        
        # Draw Results
        output_image = img_resized.copy()
        for x1, y1, x2, y2 in clean_lines:
            cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 4) 

        # Plot in the grid
        axes[idx].imshow(output_image)
        axes[idx].set_title(f"{filename} | Wires: {len(clean_lines)}")
        axes[idx].axis('off')

    # Turn off any empty subplots if num_images isn't a multiple of cols
    for i in range(len(selected_images), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Point this to whichever folder you want to batch test!
    # By default, let's test 6 random images from the PLDM test set.
    test_directory = "data/PLDM Dataset/test" 
    
    run_batch_inference(test_directory, num_images=6)