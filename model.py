# model.py
import torch.nn as nn
import torchvision.models as models

class SimpleMedNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleMedNet, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
