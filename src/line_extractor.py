import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_slope_and_intercept(x1, y1, x2, y2):
    """Calculates slope and y-intercept of a line segment."""
    if x2 - x1 == 0: # Vertical line
        return float('inf'), x1
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept

def merge_lines(lines, slope_tolerance=0.2, intercept_tolerance=50):
    """Groups and merges overlapping line segments."""
    if lines is None:
        return []

    merged_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope, intercept = get_slope_and_intercept(x1, y1, x2, y2)
        
        added_to_group = False
        
        for group in merged_lines:
            g_slope, g_intercept = group['slope'], group['intercept']
            
            if (abs(slope - g_slope) < slope_tolerance or (slope == float('inf') and g_slope == float('inf'))) and \
               abs(intercept - g_intercept) < intercept_tolerance:
               
                group['points'].extend([(x1, y1), (x2, y2)])
                group['slope'] = (group['slope'] + slope) / 2 if slope != float('inf') else float('inf')
                group['intercept'] = (group['intercept'] + intercept) / 2
                
                added_to_group = True
                break
                
        if not added_to_group:
            merged_lines.append({
                'slope': slope,
                'intercept': intercept,
                'points': [(x1, y1), (x2, y2)]
            })

    final_lines = []
    for group in merged_lines:
        points = group['points']
        if group['slope'] == float('inf') or abs(group['slope']) > 1:
            points.sort(key=lambda p: p[1])
        else:
            points.sort(key=lambda p: p[0])
            
        start_point = points[0]
        end_point = points[-1]
        final_lines.append((start_point[0], start_point[1], end_point[0], end_point[1]))

    return final_lines

def process_drone_frame(image_path, mask_path):
    """Runs the full classical CV pipeline on a simulated CNN mask."""
    print(f"Loading Image: {image_path}")
    print(f"Loading Mask: {mask_path}")
    
    img_color = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img_color is None or mask is None:
        print("Error: Could not load images. Please check the paths.")
        return

    # 1. Threshold and Thinning (Skeletonization)
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    thinned_mask = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # 2. Extract Raw Hough Lines
    raw_lines = cv2.HoughLinesP(thinned_mask, 1, np.pi/180, 40, minLineLength=40, maxLineGap=20)

    # 3. Merge Overlapping Lines
    clean_lines = merge_lines(raw_lines)

    # 4. Draw Results
    output_image = img_color.copy()
    for x1, y1, x2, y2 in clean_lines:
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 3) # Draw in thick Red

    # 5. Visualize side-by-side
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(binary_mask, cmap='gray')
    axes[0].set_title("1. CNN Mask (Ground Truth)")
    
    axes[1].imshow(thinned_mask, cmap='gray')
    axes[1].set_title("2. Skeletonization")
    
    axes[2].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f"3. Final Output ({len(clean_lines)} Wires Detected)")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Pointing to your specific relative file structure
    img_dir = "data/PLDM Dataset/train/aug_data/0.0_0"
    mask_dir = "data/PLDM Dataset/train/aug_gt/0.0_0"
    
    # Automatically find the first .jpg image in the folder
    image_files = glob.glob(os.path.join(img_dir, "*.jpg"))
    
    if not image_files:
        print(f"No .jpg files found in {img_dir}. Please make sure the dataset is extracted correctly.")
    else:
        # Grab the first image found
        target_image_path = image_files[0]
        
        # Get just the filename (e.g., "000001.jpg") and extract the base name ("000001")
        base_filename = os.path.basename(target_image_path)
        filename_without_ext = os.path.splitext(base_filename)[0]
        
        # Construct the path to the matching .png mask
        target_mask_path = os.path.join(mask_dir, f"{filename_without_ext}.png")
        
        process_drone_frame(target_image_path, target_mask_path)