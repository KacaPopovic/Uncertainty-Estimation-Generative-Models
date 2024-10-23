import torch
import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.num_classes = num_classes

        # Feature extractor layers with dropout
        self.feature_extractor = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: 32 x 28 x 28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 14 x 14
            nn.Dropout(0.25),

            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: 64 x 14 x 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 7 x 7
            nn.Dropout(0.25),

            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Output: 128 x 7 x 7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Output: 128 x 1 x 1
            nn.Dropout(0.25)
        )

        # Fully connected layer for classification
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(x.size(0), -1)  # Flatten to (batch_size, 128)

        # Classify
        logits = self.classifier(features)

        # Return both logits and features
        return logits, features
