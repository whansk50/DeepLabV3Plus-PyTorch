import torch

#calculating iou each iteration
def iou(pred, target, n_classes):
    ious = []
    for cls in range(n_classes):
        clstarget = target[...,cls]
        clspred = pred[...,cls]
        intersection = torch.logical_and(clstarget, clspred)
        union = torch.logical_or(clstarget, clspred)
        iou_score = torch.count_nonzero(intersection) / torch.count_nonzero(union)
        ious.append(iou_score)
    ious = torch.tensor(ious)
    return ious