import torch
import torch.nn as nn
from torchvision.models import resnet50
import timm

class ResNetViTLateFusion(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # output: 2048-dim features

        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()  # output: 768-dim features

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 768, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        resnet_feats = self.resnet(x)
        vit_feats = self.vit(x)
        combined = torch.cat([resnet_feats, vit_feats], dim=1)
        out = self.classifier(self.dropout(combined))
        return combined, out

def set_trainable_layers(model, freeze_until_vit=4, freeze_resnet=True):
    for param in model.resnet.parameters():
        param.requires_grad = not freeze_resnet

    for param in model.vit.parameters():
        param.requires_grad = False

    for block in model.vit.blocks[-freeze_until_vit:]:
        for param in block.parameters():
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.dropout.parameters():
        param.requires_grad = True
