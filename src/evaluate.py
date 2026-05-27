import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Import our dataset logic
from dataset import PowerlineDataset

def calculate_confusion_matrix_elements(pred, true):
    # Flatten tensors to 1D arrays for pixel-wise comparison
    pred = pred.view(-1)
    true = true.view(-1)
    
    # Calculate TP, FP, TN, FN
    tp = torch.sum((pred == 1) & (true == 1)).item()
    fp = torch.sum((pred == 1) & (true == 0)).item()
    tn = torch.sum((pred == 0) & (true == 0)).item()
    fn = torch.sum((pred == 0) & (true == 1)).item()
    
    return tp, fp, tn, fn

def evaluate_model(model_path="best_drone_wire_model.pth", data_root="data/Large_Datasets", dataset_name="PLDM", batch_size=4, threshold=0.25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on {device.upper()}...")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train the model first.")
        return
        
    # 1. Load Dataset
    test_dataset = PowerlineDataset(root_dir=data_root, dataset_name=dataset_name, split='test', img_size=(480, 640))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Testing on {len(test_dataset)} images from {dataset_name}...")
    
    # 2. Load Model
    model = smp.Unet(encoder_name="mobilenet_v2", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Accumulate Metrics
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    
    eval_loop = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for images, masks in eval_loop:
            images = images.to(device)
            masks = masks.to(device)
            
            # Predict
            raw_pred = model(images)
            probs = torch.sigmoid(raw_pred)
            
            # Thresholding
            preds = (probs > threshold).float()
            
            # Ensure ground truth is binary 0/1
            masks_binary = (masks > 0.5).float()
            
            # Calculate elements for this batch
            tp, fp, tn, fn = calculate_confusion_matrix_elements(preds, masks_binary)
            
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            
    # 4. Compute Final Metrics
    # Add epsilon to prevent division by zero
    eps = 1e-7
    
    precision = total_tp / (total_tp + total_fp + eps)
    recall = total_tp / (total_tp + total_fn + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)
    
    iou_wire = total_tp / (total_tp + total_fp + total_fn + eps)
    iou_bg = total_tn / (total_tn + total_fp + total_fn + eps)
    miou = (iou_wire + iou_bg) / 2
    
    print("\n" + "="*40)
    print("      EVALUATION RESULTS")
    print("="*40)
    print(f"Dataset:       {dataset_name}")
    print(f"Threshold:     {threshold}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1 Score:      {f1_score:.4f}")
    print(f"IoU (Wire):    {iou_wire:.4f}")
    print(f"IoU (Bg):      {iou_bg:.4f}")
    print(f"mIoU:          {miou:.4f}")
    print("="*40)
    
    # 5. Plot Confusion Matrix
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    cm = np.array([[total_tn, total_fp], [total_fn, total_tp]])
    cm_percent = cm / np.sum(cm) * 100
    
    annot_data = np.empty_like(cm_percent, dtype=object)
    for i in range(2):
        for j in range(2):
            annot_data[i, j] = f"{cm_percent[i, j]:.3f}%"
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=annot_data, fmt='', cmap='Blues', 
                xticklabels=['Predicted Background', 'Predicted Wire'],
                yticklabels=['Actual Background', 'Actual Wire'])
    plt.title(f'Pixel-Level Confusion Matrix ({dataset_name}) - Percentages')
    
    cm_path = os.path.join(output_dir, f"confusion_matrix_{dataset_name}.png")
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Confusion matrix plot saved to {cm_path}")

if __name__ == "__main__":
    # You can easily switch this to 'PLDU' to test the other dataset
    evaluate_model(dataset_name="PLDM", threshold=0.25)
