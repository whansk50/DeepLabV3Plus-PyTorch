import torch.nn as nn
import torch.nn.functional as F
import torchvision
    
class DiceFocalLoss(nn.Module):
    def __init__(self):
        super(DiceFocalLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.):
        #processed sigmoid in focal loss
        Focal = torchvision.ops.sigmoid_focal_loss(inputs=inputs, targets=targets, alpha=0.1, reduction='mean')

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        DiceFocal = dice_loss+Focal

        return DiceFocal