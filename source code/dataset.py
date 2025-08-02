import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

def load_and_encode_labels(csv_path, label_encoder=None):
    df = pd.read_csv(csv_path)
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df['label'] = label_encoder.fit_transform(df['style'])
    else:
        df['label'] = label_encoder.transform(df['style'])
    class_names = label_encoder.classes_.tolist()
    return df, class_names, label_encoder

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
