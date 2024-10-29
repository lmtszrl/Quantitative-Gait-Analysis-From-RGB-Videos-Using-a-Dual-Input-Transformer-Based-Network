import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from model import GaitModel
import torch.nn.functional as F
from torchvision import transforms


class EnsembleModel(nn.Module):
    def __init__(self):
        super(EnsembleModel, self).__init__()

        # Initialize the GaitModel
        self.gait_model = GaitModel()

        # Initialize a Vision Transformer model
        self.vit_model = vit_b_16(weights='DEFAULT')  # Use 'weights' parameter

        # Adjust the head of the ViT model
        vit_output_size = self.vit_model.heads[-1].in_features
        self.vit_model.heads = nn.Linear(vit_output_size, 128)  # Expand ViT output to 128 features

        # Define a transform to resize the input images for ViT
        self.vit_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()  # Ensure images are in tensor format
        ])

        # Expand the GaitModel output as well
        self.gait_fc = nn.Linear(1, 128)  # Expand Gait output to 128 features

        # Fully connected layers for combining outputs
        self.fc1 = nn.Linear(128 * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, img1, img2):
        # Move inputs to the same device as the model
        device = next(self.parameters()).device
        img1 = img1.to(device)
        img2 = img2.to(device)

        # Get outputs from GaitModel and ensure it's 2D
        gait_output = self.gait_model(img1, img2).unsqueeze(1)  # Shape (batch_size, 1)
        gait_output = F.relu(self.gait_fc(gait_output))

        # Process img1 for ViT model using the transform on the whole batch
        img1_vit = self.vit_transform(img1)  # Resizing for the entire batch

        # Pass the entire batch through ViT
        vit_output = self.vit_model(img1_vit)  # Output should be (batch_size, 128)

        # Reshape outputs to ensure they are 2D
        gait_output = gait_output.squeeze(1)  # Shape (batch_size, 128)

        # Check dimensions for debugging (uncomment if needed)
        # print("Gait output shape:", gait_output.shape)
        # print("ViT output shape:", vit_output.shape)

        # Concatenate the outputs (batch_size, 128*2)
        combined_output = torch.cat((gait_output, vit_output), dim=1)

        # Pass through fully connected layers
        x = F.relu(self.fc1(combined_output))  # Now it accepts a tensor of shape (batch_size, 128*2)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        output = self.fc4(x)

        return output
