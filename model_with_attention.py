import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, channels)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        se = se.view(batch_size, channels, 1, 1)
        return x * se

class GaitModelAttention(nn.Module):
    def __init__(self):
        super(GaitModelAttention, self).__init__()

        # Define the convolutional layers with SE block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 3 channels (RGB)
        self.se1 = SEBlock(32)  # SE block for attention
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.se2 = SEBlock(64)  # SE block for attention
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.se3 = SEBlock(128)  # SE block for attention
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

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
        # Pass img1 through convolutional layers with SE blocks
        x1 = self.pool1(F.relu(self.conv1(img1)))
        x1 = self.se1(x1)
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = self.se2(x1)
        x1 = self.pool3(F.relu(self.conv3(x1)))
        x1 = self.se3(x1)

        # Pass img2 through convolutional layers with SE blocks
        x2 = self.pool1(F.relu(self.conv1(img2)))
        x2 = self.se1(x2)
        x2 = self.pool2(F.relu(self.conv2(x2)))
        x2 = self.se2(x2)
        x2 = self.pool3(F.relu(self.conv3(x2)))
        x2 = self.se3(x2)

        # Flatten the features
        x1 = x1.view(x1.size(0), -1)  # Flatten
        x2 = x2.view(x2.size(0), -1)  # Flatten

        # Combine features from img1 and img2
        combined_features = torch.cat((x1, x2), dim=1)  # Shape: [batch_size, combined_feature_size]

        # Pass combined features through fully connected layers
        output = self.fc_layers(combined_features)
        return output

# Weight initialization (Xavier initialization)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
