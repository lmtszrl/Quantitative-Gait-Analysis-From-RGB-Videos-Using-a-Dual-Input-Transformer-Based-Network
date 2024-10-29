import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Define a simple model structure (e.g., a few convolutional layers)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 32 * 32, 1)  # Assuming input images are 128x128

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        return self.fc(x)

class NewEnsembleModel(nn.Module):
    def __init__(self, models):
        super(NewEnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, images1, images2):
        # Forward pass through each model and collect outputs
        outputs1 = [model(images1) for model in self.models]
        outputs2 = [model(images2) for model in self.models]

        # Average the outputs (or you could apply other strategies)
        avg_output1 = torch.mean(torch.stack(outputs1), dim=0)
        avg_output2 = torch.mean(torch.stack(outputs2), dim=0)

        return (avg_output1 + avg_output2) / 2  # Combining outputs

def create_new_ensemble_model(model_count):
    """Create an ensemble model by initializing multiple base models."""
    models = [BaseModel() for _ in range(model_count)]
    return NewEnsembleModel(models)
