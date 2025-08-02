import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix

def show_random_images_grid(df, class_names, image_dir, num_images=5, dataset_name="Dataset", val_transforms=None):
    sampled_df = df.sample(n=num_images, random_state=random.randint(0, 9999))
    plt.figure(figsize=(15, 3))
    for i, (_, row) in enumerate(sampled_df.iterrows()):
        file_path = os.path.join(image_dir, row['filename'])
        label_index = row['label']
        label_name = class_names[label_index]
        try:
            img = Image.open(file_path).convert('RGB')
            if val_transforms:
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

def plot_confusion_matrix_func(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
