import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PowerlineDataset(Dataset):
    def __init__(self, root_dir, img_size=(480, 640)):
        """
        Dynamically crawls the PLD dataset structure to pair images and masks.
        Args:
            root_dir (str): Path to the main 'data' folder containing PLDM and PLDU.
            img_size (tuple): The target size to resize images to (Height, Width).
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.pairs = []
        
        print("Crawling dataset directories. This might take a few seconds...")
        
        # Recursively find ALL .jpg files inside the root directory
        search_pattern = os.path.join(root_dir, '**', '*.jpg')
        all_jpgs = glob.glob(search_pattern, recursive=True)
        
        for img_path in all_jpgs:
            # We only want to train on the augmented training data, not the test set
            if 'aug_data' in img_path:
                
                # --- The Magic Path Swap ---
                # 1. Replace 'aug_data' with 'aug_gt' in the directory string. 
                # This perfectly handles 'aug_data_scale_0.5' -> 'aug_gt_scale_0.5'
                mask_path = img_path.replace('aug_data', 'aug_gt')
                
                # 2. Swap the file extension from .jpg to .png
                mask_path = os.path.splitext(mask_path)[0] + '.png'
                
                # 3. Verify the mask actually exists on the hard drive before adding
                if os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))
                else:
                    print(f"Warning: Missing mask for {img_path}")

        print(f"Successfully paired {len(self.pairs)} image-mask combinations.")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # 1. Read images
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Resize
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 3. Normalize to 0.0 - 1.0 for the Neural Network
        image = image.astype(np.float32) / 255.0
        
        # Ensure mask is strictly binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        
        # 4. Convert to PyTorch Tensors (Channels, Height, Width)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) 
        
        return image_tensor, mask_tensor

# --- QUICK TEST SCRIPT ---
if __name__ == "__main__":
    # Point directly to the root 'data' folder
    # Assuming you run this from the 'drone_wire_detection' root directory
    data_folder = "data" 
    
    dataset = PowerlineDataset(root_dir=data_folder, img_size=(480, 640))
    
    if len(dataset) > 0:
        img, mask = dataset[0]
        print(f"\nExample Data:")
        print(f"Image Tensor Shape: {img.shape}")
        print(f"Mask Tensor Shape:  {mask.shape}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nPyTorch is using device: {device.upper()}")
        
        if device == "cuda":
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")