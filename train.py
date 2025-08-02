from config import set_seed, DEVICE, IMAGE_DIR, EPOCHS_STAGE1, EPOCHS_STAGE2, BATCH_SIZE, IMG_SIZE
from dataset import load_and_encode_labels, ImageDataset
from transforms import get_train_transforms, get_val_transforms
from model import ResNetViTLateFusion, set_trainable_layers
from training import train_one_epoch, validate, EarlyStopping
from losses import SupConLoss, LabelSmoothingCrossEntropy
from visualize import show_random_images_grid, plot_confusion_matrix_func
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import os

def run_training():
    set_seed()

    # Load data
    train_df, class_names, label_encoder = load_and_encode_labels('train_split.csv')
    val_df, _, _ = load_and_encode_labels('val_split.csv', label_encoder=label_encoder)

    train_dataset = ImageDataset(train_df, IMAGE_DIR, transform=get_train_transforms(IMG_SIZE))
    val_dataset = ImageDataset(val_df, IMAGE_DIR, transform=get_val_transforms(IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Show some samples (optional)
    show_random_images_grid(train_df, class_names, IMAGE_DIR, num_images=5, dataset_name="Train", val_transforms=get_val_transforms(IMG_SIZE))
    show_random_images_grid(val_df, class_names, IMAGE_DIR, num_images=5, dataset_name="Validation", val_transforms=get_val_transforms(IMG_SIZE))

    model = ResNetViTLateFusion(num_classes=len(class_names)).to(DEVICE)

    early_stopping = EarlyStopping(patience=5)
    ce_loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    supcon_loss_fn = SupConLoss()

    # Stage 1
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

    print("\nLoading best model from Stage 1 before Stage 2 begins...")
    model.load_state_dict(torch.load('best_model.pth'))

    # Stage 2
    print("\nStarting Stage 2 training: Unfreeze ResNet and all ViT blocks")
    set_trainable_layers(model, freeze_until_vit=12, freeze_resnet=False)

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

    for epoch in range(min(EPOCHS_STAGE2, 5)):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, ce_loss_fn, supcon_loss_fn, DEVICE, alpha=0.5)
        val_loss, val_acc, _, _ = validate(model, val_loader, ce_loss_fn, DEVICE)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{min(EPOCHS_STAGE2,5)} - Train loss: {train_loss:.4f} - Train acc: {train_acc:.4f} - Val loss: {val_loss:.4f} - Val acc: {val_acc:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load('best_model.pth'))

    _, _, true_labels, pred_labels = validate(model, val_loader, ce_loss_fn, DEVICE)
    plot_confusion_matrix_func(true_labels, pred_labels, class_names)

    torch.save(model.state_dict(), 'final_model.pth')
    print("Training complete and model saved as final_model.pth")

if __name__ == "__main__":
    run_training()
s