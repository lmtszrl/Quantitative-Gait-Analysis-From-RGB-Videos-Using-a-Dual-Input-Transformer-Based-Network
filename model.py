import torch
import torch.nn as nn
import torch.nn.functional as F

class GaitModel(nn.Module):
    def __init__(self):
        super(GaitModel, self).__init__()

        # Define the convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: 3 channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 channels

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 channels

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 128 channels
        )

        # Calculate the size of the output from the convolutional layers
        self.conv_output_size = 128 * 16 * 16  # Assuming input image size is 128x128

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size * 2, 512),  # Combined features from both images
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization

            nn.Linear(64, 1)  # Output: one value for the target variable
        )

    def forward(self, img1, img2):
        # Pass images through convolutional layers
        features1 = self.conv_layers(img1)
        features2 = self.conv_layers(img2)

        # Flatten the features
        features1 = features1.view(features1.size(0), -1)  # Flatten
        features2 = features2.view(features2.size(0), -1)  # Flatten

        # Combine features
        combined_features = torch.cat((features1, features2), dim=1)  # Shape: [batch_size, combined_feature_size]

        # Pass combined features through fully connected layers
        output = self.fc_layers(combined_features)
        return output

# Weight initialization (Xavier initialization)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)

