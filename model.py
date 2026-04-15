import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18_Model(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        freeze_backbone=True,
        fine_tune_layers=["layer4"]
    ):
        super(ResNet18_Model, self).__init__()

        # Load pretrained (new API)
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            self.backbone = models.resnet18(weights=None)

        # Freeze toàn bộ backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Fine-tune các layer chỉ định
        for layer_name in fine_tune_layers:
            layer = getattr(self.backbone, layer_name)
            for param in layer.parameters():
                param.requires_grad = True

        # Lấy số input của FC
        in_features = self.backbone.fc.in_features

        # Thay head (có thêm dropout chống overfit)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ResNet18_Model()

    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    print("Output shape:", y.shape)  # expect [1, 2]
