import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os


class DualTargetDataset(Dataset):
    def __init__(self, csv_file, img_dir, legs_dir, transform=None, target='KneeFlex_maxExtension', subset='train', side='L'):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.legs_dir = legs_dir
        self.transform = transform
        self.target = target
        self.subset = subset
        self.side = side

        # Filter based on the subset and the side
        self.data = self.data[(self.data['dataset'] == self.subset) & (self.data['side'] == self.side)]

        # Filter out rows where images might not exist
        self.data = self.data[self.data.apply(lambda row: os.path.exists(os.path.join(self.img_dir, f"{row['videoid']}_0.png")) and
                                                os.path.exists(os.path.join(self.legs_dir, f"{row['videoid']}_{self.side}.jpg")), axis=1)]

        # Check if dataset is empty
        if self.data.empty:
            raise ValueError("No valid samples found in the dataset. Please check your image paths and CSV file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load main image
        img_path1 = os.path.join(self.img_dir, f"{self.data.iloc[idx]['videoid']}_0.png")
        image1 = Image.open(img_path1)

        # Load side-specific image
        img_path2 = os.path.join(self.legs_dir, f"{self.data.iloc[idx]['videoid']}_{self.side}.jpg")
        image2 = Image.open(img_path2)

        # Apply transformation if provided
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        # Extract the target value
        target_value = self.data.iloc[idx][self.target]
        target_tensor = torch.tensor(target_value, dtype=torch.float32)

        return image1, image2, target_tensor  # Return images and target tensor


def collate_fn(batch):
    # Filter out None values and handle the case where some samples might be missing
    batch = [b for b in batch if b is not None]
    if not batch:
        return (torch.empty(0), torch.empty(0), torch.empty(0))  # Return empty tensors if all samples are missing

    images1, images2, labels = zip(*batch)  # Unpack the batch into three lists

    # Convert lists of images and labels into tensors
    images1 = torch.stack(images1)  # Stack images1 into a tensor
    images2 = torch.stack(images2)  # Stack images2 into a tensor
    labels = torch.stack(labels)      # Stack labels into a tensor

    return images1, images2, labels


def load_data(csv_file, img_dir, legs_dir, batch_size=32, transform=None, target='KneeFlex_maxExtension', side='L'):
    if side not in ['L', 'R']:
        raise ValueError("Invalid side. Please choose 'L' or 'R'.")

    # Create datasets for the specified side
    train_dataset = DualTargetDataset(csv_file, img_dir, legs_dir, transform=transform, target=target, subset='train', side=side)
    val_dataset = DualTargetDataset(csv_file, img_dir, legs_dir, transform=transform, target=target, subset='validation', side=side)
    test_dataset = DualTargetDataset(csv_file, img_dir, legs_dir, transform=transform, target=target, subset='test', side=side)

    # Check if datasets have valid samples
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("One or more datasets are empty. Please check your data.")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
