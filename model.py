import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18_Model(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=True):
        super(ResNet18_Model, self).__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=pretrained)

        # Freeze backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Fine-tune layer4
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet18_Model()

    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print("Output shape:", y.shape)  
