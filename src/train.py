import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from tqdm import tqdm
import os

# Import the dataset class we just built
from dataset import PowerlineDataset 

# --- Hyperparameters ---
BATCH_SIZE = 4       # Start with 4. If your GPU handles it, you can bump to 8.
EPOCHS = 5           # We'll start with 5 epochs just to prove the pipeline works.
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data"   # Pointing to your root data folder
MODEL_SAVE_PATH = "best_drone_wire_model.pth"

def main():
    print(f"Initializing Training Pipeline on {DEVICE.upper()}...")

    # 1. Load the Dataset
    full_dataset = PowerlineDataset(root_dir=DATA_ROOT, img_size=(480, 640))
    
    # 2. Split into Training (90%) and Validation (10%)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {train_size} images, Validating on {val_size} images.")

    # 3. Create DataLoaders (These feed the GPU in batches)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. Initialize the U-Net Model
    # We use MobileNetV2 as the backbone because it is incredibly fast on edge devices.
    model = smp.Unet(
        encoder_name="mobilenet_v2", 
        encoder_weights="imagenet", # Pre-trained on basic objects to speed up learning
        in_channels=3,              # RGB images have 3 channels
        classes=1                   # We only want 1 output channel (Wire vs Background)
    ).to(DEVICE)

    # 5. Define Loss Function and Optimizer
    # BCEWithLogitsLoss is the mathematical standard for Binary Classification
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. The Training Loop
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # -- TRAINING PHASE --
        model.train()
        train_loss = 0.0
        
        # tqdm creates a nice progress bar in the terminal
        train_loop = tqdm(train_loader, desc="Training")
        for images, masks in train_loop:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            # Forward Pass
            predictions = model(images)
            loss = criterion(predictions, masks)

            # Backward Pass (Learn from mistakes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # -- VALIDATION PHASE --
        # We test the model on images it hasn't seen yet to ensure it isn't just memorizing
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad(): # Don't update weights during validation
            val_loop = tqdm(val_loader, desc="Validation")
            for images, masks in val_loop:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                
                predictions = model(images)
                loss = criterion(predictions, masks)
                
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # -- SAVE THE BEST MODEL --
        if avg_val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model!")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("\nTraining Complete! Best model saved to:", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()