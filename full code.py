import os
import re
import argparse
import pandas as pd
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet18_Weights, ResNet34_Weights
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ========== Dataset Classes Definition ==========
# Dataset classes for easier access and data loading
class SpongeBobDataset(Dataset):
    def __init__(self, annotations, data_dir, transform=None, preload=False, is_training=True):
        self.annotations = annotations.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        self.preload = preload
        self.images = []
        self.labels = []
        self.is_training = is_training
        
        # Character color features for attention
        self.color_features = {
            'spongebob': np.array([255, 255, 0]),  # Yellow (spongebob)
            'patrick': np.array([255, 182, 193]),  # Pink   (patrick)
            'squidward': np.array([64, 224, 208])  # Teal   (squidward)
        }
        
        if self.preload:
            print("Preloading images...")
            for idx in range(len(self.annotations)):
                rel = self.annotations.loc[idx, 'image_path'].replace("\\", "/")
                img_path = os.path.join(self.data_dir, rel)
                try:
                    image = Image.open(img_path).convert("RGB")
                    self.images.append(image)
                    
                    label = np.zeros(3)
                    label[0] = self.annotations.loc[idx, 'spongebob']
                    label[1] = self.annotations.loc[idx, 'squidward']
                    label[2] = self.annotations.loc[idx, 'patrick']
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            print("Preloading complete")

    def __len__(self):
        return len(self.annotations)

    # Create attention masks based on character colors
    def create_color_mask(self, img_array):
        # Convert PIL to numpy if needed
        if isinstance(img_array, Image.Image):
            img_array = np.array(img_array)
        
        # Initialize masks
        masks = np.zeros((3, img_array.shape[0], img_array.shape[1]))
        
        # Create color-based masks for each character
        for i, (char, color) in enumerate(self.color_features.items()):
            # Color distance for each pixel
            diff = np.sqrt(np.sum((img_array - color.reshape(1, 1, 3))**2, axis=2))
            # Normalize and invert (closer = higher value)
            mask = np.clip(1.0 - diff / diff.max(), 0, 1)
            masks[i] = mask
            
        return masks

   
    def __getitem__(self, idx):
        # Load image and label
        if self.preload:
            image = self.images[idx]
            label = self.labels[idx]
        else:
            rel = self.annotations.loc[idx, 'image_path'].replace("\\", "/")
            img_path = os.path.join(self.data_dir, rel)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
            label = np.array([
                self.annotations.loc[idx, 'spongebob'],
                self.annotations.loc[idx, 'squidward'],
                self.annotations.loc[idx, 'patrick']
            ], dtype=np.float32)
    
        # Create raw mask (original size)
        raw_masks = torch.from_numpy(self.create_color_mask(image)).float()  # [3, H0, W0]
    
        # Apply transform to image
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=np.array(image))
                image = augmented['image']      # Tensor [3, H, W]
            else:
                image = self.transform(image)   # Tensor [3, H, W]
    
        # Resize masks to match transformed image size
        #    image.shape[1:] is (H, W)
        color_masks = F.interpolate(
            raw_masks.unsqueeze(0),            # [1, 3, H0, W0]
            size=image.shape[1:],              # (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)                           # back to [3, H, W]
    
        # Return image, label, and masks if training
        if self.is_training:
            return image, torch.from_numpy(label), color_masks
        else:
            return image, torch.from_numpy(label)

# Test class
class SpongeBobTestDataset(Dataset):
    def __init__(self, file_list, data_dir, transform=None, preload=False):
        self.file_list = [p.replace("\\", "/") for p in file_list]
        self.data_dir = data_dir
        self.transform = transform
        self.preload = preload
        self.images = []
        
        # Character color features for attention
        self.color_features = {
            'spongebob': np.array([255, 255, 0]),  # Yellow
            'patrick': np.array([255, 182, 193]),  # Pink
            'squidward': np.array([64, 224, 208])  # Teal
        }
        
        if self.preload:
            print("Preloading test images...")
            for rel in self.file_list:
                img_path = os.path.join(self.data_dir, rel)
                try:
                    image = Image.open(img_path).convert("RGB")
                    self.images.append(image)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    # Use a black image as fallback
                    self.images.append(Image.new('RGB', (224, 224), (0, 0, 0)))
            print("Preloading complete")

    def __len__(self):
        return len(self.file_list)

    # Create attention masks based on character colors
    def create_color_mask(self, img_array):
        # Convert PIL to numpy if needed
        if isinstance(img_array, Image.Image):
            img_array = np.array(img_array)
        
        # Initialize masks
        masks = np.zeros((3, img_array.shape[0], img_array.shape[1]))
        
        # Create color-based masks for each character
        for i, (char, color) in enumerate(self.color_features.items()):
            # Color distance for each pixel
            diff = np.sqrt(np.sum((img_array - color.reshape(1, 1, 3))**2, axis=2))
            # Normalize and invert (closer = higher value)
            mask = np.clip(1.0 - diff / diff.max(), 0, 1)
            masks[i] = mask
            
        return masks

    def __getitem__(self, idx):
        rel = self.file_list[idx]
    
        # Load image (preloaded or from disk)
        if self.preload:
            image = self.images[idx]
        else:
            img_path = os.path.join(self.data_dir, rel)
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.new('RGB', (224, 224), (0, 0, 0))
    
        # Compute raw color masks at original size
        raw_masks = torch.from_numpy(self.create_color_mask(image)).float()  # [3, H0, W0]
    
        # Apply transforms to the image
        if self.transform:
            if isinstance(self.transform, A.Compose):
                augmented = self.transform(image=np.array(image))
                image = augmented['image']  # Tensor [3, H, W]
            else:
                image = self.transform(image)  # Tensor [3, H, W]
    
        # Interpolate masks to match transformed image spatial dims
        color_masks = F.interpolate(
            raw_masks.unsqueeze(0),       # [1, 3, H0, W0]
            size=image.shape[1:],         # (H, W)
            mode='bilinear',
            align_corners=False
        ).squeeze(0)                     # [3, H, W]
    
        # Return image, masks, and path
        return image, color_masks, rel


# ========== Data Transformations ==========
# Get data transformation
def get_transforms(img_size=224):
    norm_args = dict(mean=[0.485, 0.456, 0.406],
                     std =[0.229, 0.224, 0.225])

    train_transform = A.Compose([
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.5, 1.0),
            ratio=(3/4, 4/3),
            interpolation=cv2.INTER_LINEAR,
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2,
            p=0.7
        ),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.5
        ),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=10,
            p=0.5
        ),
        A.Normalize(**norm_args),
        ToTensorV2(),
    ])

    resized = int(img_size * 1.14)
    valid_transform = A.Compose([
        A.Resize(height=resized, width=resized, p=1.0),
        A.CenterCrop(height=img_size, width=img_size, p=1.0),
        A.Normalize(**norm_args),
        ToTensorV2(),
    ])

    return train_transform, valid_transform


# ====== Color Attention Module and Model Definition ======
class ColorAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ColorAttentionModule, self).__init__()
        # Process color masks to create attention weights
        self.conv = nn.Conv2d(3, in_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, color_masks):
        # Resize color masks to match feature map size
        b, c, h, w = x.size()
        if color_masks.size()[2] != h or color_masks.size()[3] != w:
            color_masks = F.interpolate(color_masks, size=(h, w), mode='bilinear', align_corners=False)
        
        # Generate attention weights from color masks
        attention = self.sigmoid(self.bn(self.conv(color_masks)))
        
        # Apply attention to feature maps
        return x * attention


class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes=3, backbone='resnet50', dropout=0.5):
        super(ResNetWithAttention, self).__init__()
        # Load backbone with weights
        weight_map = {
            'resnet18': ResNet18_Weights.DEFAULT,
            'resnet34': ResNet34_Weights.DEFAULT,
            'resnet50': ResNet50_Weights.DEFAULT
        }
        
        if backbone not in weight_map:
            raise ValueError(f"Unsupported backbone {backbone}")
        
        weights = weight_map[backbone]
        self.backbone_name = backbone
        model_fn = getattr(models, backbone)
        backbone_model = model_fn(weights=weights)
        
        # Remove the original FC layer
        self.features = nn.Sequential(*list(backbone_model.children())[:-2])
        
        # Feature dimensions based on backbone
        if backbone == 'resnet18' or backbone == 'resnet34':
            feature_dim = 512
        else:
            feature_dim = 2048
        
        # Color attention module
        self.color_attention = ColorAttentionModule(feature_dim)
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # FC layers with multi-level dropout for regularization
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(feature_dim, feature_dim//2)
        self.ln1 = nn.LayerNorm(feature_dim//2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout/2)
        self.fc2 = nn.Linear(feature_dim//2, num_classes)
        
    def forward(self, x, color_masks=None):
        # Extract features from backbone
        feat = self.features(x)
        
        # Apply color attention if masks are provided
        if color_masks is not None:
            feat = self.color_attention(feat, color_masks)
        
        # Global average pooling
        feat = self.gap(feat).view(x.size(0), -1)
        
        # Fully connected layers
        feat = self.dropout1(feat)
        feat = self.fc1(feat)
        feat = self.ln1(feat)
        feat = self.relu(feat)
        feat = self.dropout2(feat)
        output = self.fc2(feat)
        
        return output


# ===== Training and Validation Functions (Focal Loss for Class Imbalance) ====
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            # expect alpha as list/array of length=num_classes
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs, targets: [batch, num_classes]
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)          # [batch, num_classes]
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * bce       # [batch, num_classes]

        if self.alpha is not None:
            # alpha: [num_classes]; expand to [batch, num_classes]
            alpha = self.alpha.to(inputs.device).unsqueeze(0)
            loss = loss * alpha       # broadcast per-class

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, epoch=0, epochs=1):
    model.train()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_targets = []
    all_outputs = []
    start_time = time.time()
    
    for batch_idx, (images, targets, color_masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        color_masks = color_masks.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast():
            outputs = model(images, color_masks)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        
        # For binary outputs
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_targets.append(targets.cpu())
        all_outputs.append(probs.detach().cpu())
        
        # Calculate correct predictions for each class
        correct_preds += (preds == targets).sum().item()
        total_preds += targets.numel()
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}/{epochs} [{batch_idx}/{len(loader)}] '
                  f'Loss: {loss.item():.4f}')
    
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    
    epoch_time = time.time() - start_time
    epoch_acc = correct_preds / total_preds
    epoch_loss = running_loss / len(loader.dataset)
    
    return epoch_loss, epoch_acc, epoch_time, all_targets.numpy(), all_outputs.numpy()


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    all_targets = []
    all_outputs = []
    
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)  # No color masks for validation set
            loss = criterion(outputs, targets).item() * images.size(0)
        
        running_loss += loss
        
        # For binary outputs
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        
        all_targets.append(targets.cpu())
        all_outputs.append(probs.cpu())
        
        # Calculate correct predictions for each class
        correct_preds += (preds == targets).sum().item()
        total_preds += targets.numel()
    
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    
    # Calculate F1 score per class
    val_loss = running_loss / len(loader.dataset)
    val_acc = correct_preds / total_preds
    
    return val_loss, val_acc, all_targets.numpy(), all_outputs.numpy()


@torch.no_grad()
def inference(model, loader, device):
    model.eval()
    all_paths = []
    all_probs = []
    
    for images, color_masks, paths in loader:
        images = images.to(device, non_blocking=True)
        color_masks = color_masks.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images, color_masks)
            probs = torch.sigmoid(outputs)
        
        all_paths.extend(paths)
        all_probs.extend(probs.cpu().numpy())
    
    return all_paths, np.array(all_probs)


# ==================== Visualization Functions ====================
def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, f1_scores=None, save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot losses
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracies
    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(targets, predictions, classes=['SpongeBob', 'Squidward', 'Patrick'], save_path=None):
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
    plt.close()


# ===================== Visualize Predictions for Report =====================
def visualize_predictions(model, dataset, indices, device, class_names=['SpongeBob', 'Squidward', 'Patrick'], save_path=None):
    """
    Visualize model predictions on sample images
    """
    model.eval()
    fig, axes = plt.subplots(len(indices), 2, figsize=(14, 5*len(indices)))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image and target
            image, target, color_masks = dataset[idx]
            
            # Convert tensor to numpy for visualization
            img_np = image.permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)
            
            # Forward pass
            image_tensor = image.unsqueeze(0).to(device)
            color_masks_tensor = color_masks.unsqueeze(0).to(device)
            outputs = model(image_tensor, color_masks_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]
            
            # Plot image
            axes[i, 0].imshow(img_np)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Plot predictions
            target_np = target.numpy()
            bars = axes[i, 1].bar(class_names, probs)
            axes[i, 1].set_ylim(0, 1)
            axes[i, 1].set_title('Predicted Probabilities vs Ground Truth')
            axes[i, 1].set_ylabel('Probability')
            
            # Highlight ground truth with a different color
            for j, bar in enumerate(bars):
                if target_np[j] > 0.5:
                    bar.set_color('green')
                else:
                    bar.set_color('blue')
            
            # Add text for actual values
            for j, p in enumerate(probs):
                axes[i, 1].text(j, p + 0.05, f'{p:.2f}\nGT: {int(target_np[j])}', 
                         ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()


# ===================== Testing and Submission Function =====================
def test_and_submit(model_paths, test_file_list, data_dir, transform, device, output_path="submission.csv"):
    # Ensure models list is a list even if only one model is provided
    if not isinstance(model_paths, list):
        model_paths = [model_paths]
    
    # Create test dataset and dataloader
    test_ds = SpongeBobTestDataset(test_file_list, data_dir, transform=transform)
    test_loader = DataLoader(
        test_ds, 
        batch_size=32,
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    
    # Perform inference with ensemble of models
    all_predictions = []
    
    for model_path in model_paths:
        print(f"Inferring with model: {model_path}")
        # Load model
        model = ResNetWithAttention(num_classes=3, backbone='resnet50', dropout=0).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Get predictions
        paths, probs = inference(model, test_loader, device)
        all_predictions.append(probs)
    
    # Average predictions from all models (ensemble)
    final_probs = np.mean(all_predictions, axis=0)
    
    # Convert probabilities to binary predictions
    final_preds = (final_probs > 0.5).astype(int)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'image_path': paths,
        'spongebob': final_preds[:, 0],
        'squidward': final_preds[:, 1],
        'patrick': final_preds[:, 2]
    })
    
    # Save submission
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    # Calculate distribution of predictions
    print("\nPrediction Distribution:")
    print(f"SpongeBob: {submission['spongebob'].mean():.4f}")
    print(f"Squidward: {submission['squidward'].mean():.4f}")
    print(f"Patrick: {submission['patrick'].mean():.4f}")
    
    # Count instances of multiple character predictions
    multiple_chars = (submission['spongebob'] + submission['squidward'] + submission['patrick'] > 1).sum()
    print(f"Images with multiple characters: {multiple_chars} ({multiple_chars/len(submission)*100:.2f}%)")
    
    # Count instances of no character predictions
    no_chars = ((submission['spongebob'] + submission['squidward'] + submission['patrick']) == 0).sum()
    print(f"Images with no characters: {no_chars} ({no_chars/len(submission)*100:.2f}%)")
    
    return submission


# =========== Data Analysis Function ==========
def analyze_data(df):
    """Analyze the distribution of labels in the dataset"""
    print("\nDataset Distribution:")
    total = len(df)
    
    # Individual character counts
    spongebob_count = df['spongebob'].sum()
    squidward_count = df['squidward'].sum()
    patrick_count = df['patrick'].sum()
    
    print(f"SpongeBob: {spongebob_count} ({spongebob_count/total*100:.2f}%)")
    print(f"Squidward: {squidward_count} ({squidward_count/total*100:.2f}%)")
    print(f"Patrick: {patrick_count} ({patrick_count/total*100:.2f}%)")
    
    # Count instances with multiple characters
    df['character_count'] = df['spongebob'] + df['squidward'] + df['patrick']
    multi_char = (df['character_count'] > 1).sum()
    print(f"Images with multiple characters: {multi_char} ({multi_char/total*100:.2f}%)")
    
    # Count instances with no characters
    no_char = (df['character_count'] == 0).sum()
    print(f"Images with no characters: {no_char} ({no_char/total*100:.2f}%)")
    
    # Character co-occurrence
    sb_sq = ((df['spongebob'] == 1) & (df['squidward'] == 1)).sum()
    sb_pat = ((df['spongebob'] == 1) & (df['patrick'] == 1)).sum()
    sq_pat = ((df['squidward'] == 1) & (df['patrick'] == 1)).sum()
    all_three = ((df['spongebob'] == 1) & (df['squidward'] == 1) & (df['patrick'] == 1)).sum()
    
    print("\nCharacter Co-occurrence:")
    print(f"SpongeBob + Squidward: {sb_sq} ({sb_sq/total*100:.2f}%)")
    print(f"SpongeBob + Patrick: {sb_pat} ({sb_pat/total*100:.2f}%)")
    print(f"Squidward + Patrick: {sq_pat} ({sq_pat/total*100:.2f}%)")
    print(f"All three characters: {all_three} ({all_three/total*100:.2f}%)")
    
    return {
        'spongebob': spongebob_count/total,
        'squidward': squidward_count/total,
        'patrick': patrick_count/total,
        'multi_char': multi_char/total,
        'no_char': no_char/total
    }


# ===================== Main Function Loop =====================
def main(args):
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Start timer
    start_time = time.time()
    
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Set device with GPU if available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load annotations (handle different path formats)
    train_csv = args.train_csv or os.path.join(args.data_dir, 'train.csv')
    df = pd.read_csv(train_csv)
    df.iloc[:, 0] = df.iloc[:, 0].str.replace(r"\\", "/", regex=True)
    
    # Analyze the data distribution
    print("\n=== Dataset Analysis ===")
    stats = analyze_data(df)
    
    # Automatically detect test list
    file_list = []
    for root, _, files in os.walk(args.data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                rel = os.path.relpath(os.path.join(root, f), args.data_dir).replace('\\', '/')
                if re.match(r'^S01E\d{2}[abc]?/', rel):
                    file_list.append(rel)
    
    print(f"Found {len(file_list)} test images")
    
    # Get transforms
    train_transform, valid_transform = get_transforms(img_size=args.img_size)
    
    # Cross-validation with stratification by all three character labels
    n_splits = args.folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    
    # Create a stratification column for the multi-label case
    df['strat_col'] = df['spongebob'].astype(str) + df['squidward'].astype(str) + df['patrick'].astype(str)
    
    # Initialize lists to store model paths and metrics
    model_paths = []
    val_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['strat_col'])):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")
        
        # Create datasets
        train_ds = SpongeBobDataset(train_df, args.data_dir, transform=train_transform, is_training=True)
        val_ds = SpongeBobDataset(val_df, args.data_dir, transform=valid_transform, is_training=False)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_ds, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=args.batch_size*2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True
        )
        
        # Create model
        model = ResNetWithAttention(
            num_classes=3, 
            backbone=args.backbone,
            dropout=args.dropout
        ).to(device)
        
        # Use Focal Loss with class weights based on distribution
        alpha = [
            1 / (stats['spongebob'] + 1e-5),
            1 / (stats['squidward'] + 1e-5),
            1 / (stats['patrick'] + 1e-5)
        ]
        alpha = [a / sum(alpha) for a in alpha]  # Normalize weights
        
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        # Gradient scaler for mixed precision training
        scaler = GradScaler()
        
        # Lists to store metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # Training loop
        best_val_acc = 0
        best_model_path = None
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            # Train
            train_loss, train_acc, train_time, train_targets, train_outputs = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, epoch, args.epochs
            )
            
            # Validate
            val_loss, val_acc, val_targets, val_outputs = validate(
                model, val_loader, criterion, device
            )
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Time: {train_time:.2f}s")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = f"outputs/model_fold{fold}_epoch{epoch}_acc{val_acc:.4f}.pth"
                torch.save(model.state_dict(), model_path)
                best_model_path = model_path
                print(f"Saved best model with validation accuracy: {val_acc:.4f}")
                
                # Generate classification report for best model
                val_preds = (val_outputs > 0.5).astype(int)
                report = classification_report(
                    val_targets, val_preds, 
                    target_names=['SpongeBob', 'Squidward', 'Patrick'],
                    zero_division=0
                )
                print("\nClassification Report:")
                print(report)
                
                # Save confusion matrices for each class
                for i, class_name in enumerate(['SpongeBob', 'Squidward', 'Patrick']):
                    cm_path = f"outputs/cm_fold{fold}_class{i}_{class_name}.png"
                    plot_confusion_matrix(
                        val_targets[:, i], val_preds[:, i],
                        classes=[f'Not {class_name}', class_name],
                        save_path=cm_path
                    )
            
            # Update scheduler
            scheduler.step()
            
        # Store best model path and validation score
        if best_model_path:
            model_paths.append(best_model_path)
            val_scores.append(best_val_acc)
    
    # Ensemble inference and submission
    submission = test_and_submit(
        model_paths=model_paths,
        test_file_list=file_list,
        data_dir=args.data_dir,
        transform=valid_transform,
        device=device,
        output_path="submission.csv"
    )
    
    # Print total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    print(f"Best validation scores per fold: {val_scores}")
    print(f"Mean validation score: {np.mean(val_scores):.4f}")
    

if __name__ == "__main__":
    # Kaggle optimized parameters for best performance
    class Args:
        def __init__(self):
            # Kaggle paths
            self.data_dir = "/kaggle/input/spongebob-classification"
            self.train_csv = "/kaggle/input/spongebob-classification/train.csv"
            
            # Model configuration (ResNet50 performs best for this task)
            self.backbone = "resnet50"
            self.img_size = 320  # Larger image size to capture more details
            
            # Training parameters
            self.batch_size = 16  # Smaller batch size for better generalization
            self.epochs = 30      # More epochs for better convergence
            self.lr = 4e-4              # Lower learning rate for stable training
            self.weight_decay = 2e-5    # Reduced weight decay to prevent overfitting
            self.dropout = 0.4          # Optimized dropout rate
            self.focal_gamma = 2.5      # Slightly increased gamma for better handling of class imbalance (as the dataset is)
            
            # Cross-validation
            self.folds = 5    # 5-fold CV sweet spot between performance and training time
            self.seed = 2023  # Different seed for better generalization
            
            # Others
            self.workers = 2 
    
    # Run main with defined arguments
    args = Args()
    main(args)