import torch
from torch import nn

from torchvision.models.mobilenetv2 import MobileNetV2


class MobileNet_v2(nn.Module):
    def __init__(self, output_size: int):
        super().__init__()
        self.model = MobileNetV2()
        self.model.classifier = nn.Identity()

        self.dropout = nn.Dropout(p=0.2)

        self.clf = nn.Linear(1280, output_size)

    def forward(self, im: torch.Tensor) -> torch.Tensor:
        emb = self.model(im)
        emb = self.dropout(emb)

        logits = self.clf(emb)

        return logits
