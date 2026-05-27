import os
import cv2
import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset, DataLoader


class PowerlineDataset(Dataset):
    def __init__(self, root_dir, dataset_name='PLDU', split='train', img_size=(480, 640)):
        """
        Args:
            root_dir (str): Path to the main 'data/Large_Datasets' folder.
            dataset_name (str): 'PLDU' or 'PLDM'.
            split (str): 'train' or 'test'.
            img_size (tuple): The target size to resize images to (Height, Width).
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size
        self.pairs = []
        
        # Base folder for the specific dataset (PLDU or PLDM)
        base_dir = os.path.join(self.root_dir, self.dataset_name)
        
        if self.split == 'train':
            lst_file = os.path.join(base_dir, f"{self.dataset_name}_{self.split}_pair.lst")
            print(f"Loading training pairs from {lst_file}...")
            if not os.path.exists(lst_file):
                raise FileNotFoundError(f"List file not found: {lst_file}")
                
            with open(lst_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 2:
                    img_subpath, mask_subpath = parts
                    img_path = os.path.join(base_dir, img_subpath)
                    mask_path = os.path.join(base_dir, mask_subpath)
                    
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.pairs.append((img_path, mask_path))
                    else:
                        print(f"Warning: Missing file for {img_path} or {mask_path}")
                        
        elif self.split == 'test':
            lst_file = os.path.join(base_dir, f"{self.dataset_name}_{self.split}.lst")
            print(f"Loading test pairs from {lst_file}...")
            if not os.path.exists(lst_file):
                raise FileNotFoundError(f"List file not found: {lst_file}")
                
            with open(lst_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                img_subpath = line.strip()
                if not img_subpath:
                    continue
                
                img_path = os.path.join(base_dir, img_subpath)
                
                # Replace 'test/' with 'test_gt/' and '.jpg' with '.mat'
                mask_subpath = img_subpath.replace('test/', 'test_gt/')
                mask_subpath = os.path.splitext(mask_subpath)[0] + '.mat'
                mask_path = os.path.join(base_dir, mask_subpath)
                
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.pairs.append((img_path, mask_path))
                else:
                    print(f"Warning: Missing file for {img_path} or {mask_path}")
                    
        else:
            raise ValueError(f"Invalid split: {self.split}")

        print(f"Successfully paired {len(self.pairs)} image-mask combinations.")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        # 1. Read image
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Read mask depending on file extension
        if mask_path.endswith('.mat'):
            mat = scipy.io.loadmat(mask_path)
            
            if 'groundTruth' in mat:
                val = mat['groundTruth']
                try:
                    # Handle common edge-detection nested struct format
                    mask = val[0, 0]['Boundaries'][0, 0]
                except (IndexError, ValueError, KeyError, TypeError):
                    mask = val
            elif 'GT' in mat:
                mask = mat['GT']
            else:
                keys = [k for k in mat.keys() if not k.startswith('__')]
                mask = mat[keys[0]] if keys else np.zeros((self.img_size[0], self.img_size[1]), dtype=np.uint8)
                
            mask = np.array(mask)
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Failed to read mask: {mask_path}")
            
        # 3. Resize
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 4. Normalize to 0.0 - 1.0 for the Neural Network
        image = image.astype(np.float32) / 255.0
        
        # Ensure mask is strictly binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32) / 255.0
        
        # 5. Convert to PyTorch Tensors (Channels, Height, Width)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) 
        
        return image_tensor, mask_tensor

# --- QUICK TEST SCRIPT ---
if __name__ == "__main__":
    data_folder = "data/Large_Datasets" 
    
    dataset = PowerlineDataset(root_dir=data_folder, dataset_name='PLDU', split='train', img_size=(480, 640))
    
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