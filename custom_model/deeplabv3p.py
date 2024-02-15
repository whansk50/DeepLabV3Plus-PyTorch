import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_model.backbone_X import ModifiedXception
from custom_model.ASPP import ASPP    

class Decoder(nn.Module):
    def __init__(self, num_classes):
      super(Decoder, self).__init__()

      self.num_classes = num_classes

      self.conv1 = nn.Conv2d(128, 48, 1, bias=False)  # low-level feature를 처리하기 위한 1x1 convolution
      self.bn1 = nn.BatchNorm2d(48)
      self.relu = nn.ReLU()

      self.last_conv = nn.Sequential(
        nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1)
      )

    def forward(self, aspp_features, low_level_features):
        low_level_features = self.conv1(low_level_features)
        low_level_features = self.bn1(low_level_features)
        low_level_features = self.relu(low_level_features)

        aspp_features = F.interpolate(aspp_features, size=low_level_features.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat((aspp_features, low_level_features), dim=1)
        x = self.last_conv(x)
        h,w = x.shape[2], x.shape[3]
        x = F.interpolate(x, size=(h+1, w+1), mode='bilinear', align_corners=False)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x
    
class DeepLabV3_Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3_Plus, self).__init__()
        self.backbone = ModifiedXception()
        #self.aspp = ASPP(728, [12, 24, 36])
        #self.aspp = ASPP(1536, [12, 24, 36])
        self.aspp = ASPP(1536, [6, 12, 18])
        self.decoder = Decoder(num_classes)
        #self.classifier = DeepLabHead(508, num_classes)

    def forward(self, x):
        low_level_features, high_level_features = self.backbone(x)
        x = self.aspp(high_level_features)
        x = self.decoder(x, low_level_features)
        #x = self.classifier(x)
        return x