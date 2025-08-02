import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision import transforms
import timm

# === 1. Set seeds for reproducibility ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# === 2. Parameters ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_STAGE1 = 10
EPOCHS_STAGE2 = 15
IMAGE_DIR = 'raw_images'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 3. Label encoding ===
def load_and_encode_labels(csv_path, label_encoder=None):
    df = pd.read_csv(csv_path)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['style'])
    else:
        df['label'] = label_encoder.transform(df['style'])
    class_names = label_encoder.classes_.tolist()
    return df, class_names, label_encoder

# === 4. Custom Dataset ===
class ImageDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = row['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# === 5. Data augmentation and transforms ===
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ColorJitter(contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === 6. Load dataframes and datasets ===
train_df, class_names, label_encoder = load_and_encode_labels('train_split.csv')
val_df, _, _ = load_and_encode_labels('val_split.csv', label_encoder=label_encoder)

missing_files = [
    os.path.join(IMAGE_DIR, f) for f in train_df['filename']
    if not os.path.exists(os.path.join(IMAGE_DIR, f))
]
if missing_files:
    print(f"[WARNING] {len(missing_files)} missing images. First few:\n", missing_files[:5])

train_dataset = ImageDataset(train_df, IMAGE_DIR, transform=train_transforms)
val_dataset = ImageDataset(val_df, IMAGE_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# === 7. Show sample images (optional) ===
def show_random_images_grid(df, class_names, image_dir, num_images=5, dataset_name="Dataset"):
    sampled_df = df.sample(n=num_images, random_state=random.randint(0, 9999))
    plt.figure(figsize=(15, 3))
    for i, (_, row) in enumerate(sampled_df.iterrows()):
        file_path = os.path.join(image_dir, row['filename'])
        label_index = row['label']
        label_name = class_names[label_index]
        try:
            img = Image.open(file_path).convert('RGB')
            img = val_transforms(img).permute(1, 2, 0).numpy()
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.title(f"{label_name}")
            plt.axis('off')
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
    plt.suptitle(f"{dataset_name} Samples", fontsize=16)
    plt.tight_layout()
    plt.show()

show_random_images_grid(train_df, class_names, IMAGE_DIR, num_images=5, dataset_name="Train")
show_random_images_grid(val_df, class_names, IMAGE_DIR, num_images=5, dataset_name="Validation")

# === 8. Late Fusion Model: ResNet50 + ViT-B/16 ===
class ResNetViTLateFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # output: 2048-dim features

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # output: 768-dim features

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        resnet_feats = self.resnet(x)    # (batch, 2048)
        vit_feats = self.vit(x)          # (batch, 768)
        combined = torch.cat([resnet_feats, vit_feats], dim=1)
        out = self.classifier(self.dropout(combined))
        return combined, out

model = ResNetViTLateFusion(num_classes=len(class_names)).to(DEVICE)

# === Freeze/Unfreeze utility ===
def set_trainable_layers(model, freeze_until_vit=4, freeze_resnet=True):
    # Freeze or unfreeze ResNet
    for param in model.resnet.parameters():
        param.requires_grad = not freeze_resnet

    # Freeze all ViT params
    for param in model.vit.parameters():
        param.requires_grad = False

    # Unfreeze last freeze_until_vit ViT blocks
    for block in model.vit.blocks[-freeze_until_vit:]:
        for param in block.parameters():
            param.requires_grad = True

    # Always unfreeze classifier and dropout
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.dropout.parameters():
        param.requires_grad = True

# === 9. SupCon Loss ===
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        features = nn.functional.normalize(features, dim=1)
        batch_size = features.size(0)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        loss = -mean_log_prob_pos
        return loss.mean()

# === 10. Label Smoothing Loss ===
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        log_probs = self.log_softmax(x)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

ce_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
supcon_loss_fn = SupConLoss()

# === Early Stopping ===
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

early_stopping = EarlyStopping(patience=5)

# === 11. Training and Validation Functions ===
def train_one_epoch(model, dataloader, optimizer, ce_loss_fn, supcon_loss_fn, device, alpha=0.5):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        features, logits = model(images)
        loss = alpha * supcon_loss_fn(features, labels) + (1 - alpha) * ce_loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate(model, dataloader, ce_loss_fn, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, logits = model(images)
            loss = ce_loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, all_labels, all_preds

# === 12. Confusion matrix plotting function ===
def plot_confusion_matrix_func(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# === 13. Training Loop ===
# === 13. Training Loop ===
def run_training():
    print("Starting Stage 1 training: Freeze ResNet, unfreeze last 4 ViT blocks")
    set_trainable_layers(model, freeze_until_vit=4, freeze_resnet=True)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-6)

    for epoch in range(EPOCHS_STAGE1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, supcon_loss_fn, DEVICE, alpha=0.5)
        val_loss, val_acc, _, _ = validate(model, val_loader, ce_loss_fn, DEVICE)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} - Train loss: {train_loss:.4f} - Train acc: {train_acc:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    #  Load best model from Stage 1 before Stage 2 starts
    print("\nLoading best model from Stage 1 before Stage 2 begins...")
    model.load_state_dict(torch.load('best_model.pth'))

    print("\nStarting Stage 2 training: Unfreeze ResNet and all ViT blocks")
    set_trainable_layers(model, freeze_until_vit=12, freeze_resnet=False)  # unfreeze all ViT blocks + ResNet

    # Optionally, set different LR for parameter groups
    param_groups = [
        {"params": model.vit.blocks[:4].parameters(), "lr": 1e-7},
        {"params": model.vit.blocks[4:8].parameters(), "lr": 5e-7},
        {"params": model.vit.blocks[8:].parameters(), "lr": 1e-6},
        {"params": model.classifier.parameters(), "lr": 5e-6},
        {"params": model.resnet.parameters(), "lr": 1e-6},
    ]
    optimizer = optim.Adam(param_groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3, min_lr=1e-8)

    early_stopping.counter = 0
    early_stopping.early_stop = False

    for epoch in range(min(EPOCHS_STAGE2, 5)):  # limit Stage 2 to max 5 epochs or EPOCHS_STAGE2
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, supcon_loss_fn, DEVICE, alpha=0.5)
        val_loss, val_acc, _, _ = validate(model, val_loader, ce_loss_fn, DEVICE)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{min(EPOCHS_STAGE2,5)} - Train loss: {train_loss:.4f} - Train acc: {train_acc:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load('best_model.pth'))

run_training()

# === 14. Confusion Matrix and Save ===
_, _, true_labels, pred_labels = validate(model, val_loader, ce_loss_fn, DEVICE)
plot_confusion_matrix_func(true_labels, pred_labels, class_names)
torch.save(model.state_dict(), 'final_model.pth')
print("Training complete and model saved as final_model.pth")


