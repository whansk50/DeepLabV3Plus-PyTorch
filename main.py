from torch.utils.data import DataLoader
import dataloader
import torch
from torch import nn
import wandb
from tqdm import tqdm
import cfg
from custom_model import deeplabv3p
import numpy as np
from metrics import iou
import loss

cuda = torch.cuda.is_available()
DEVICE=[]

if cuda:
  for i in range(torch.cuda.device_count()):
    DEVICE.append(f"cuda:{i}")
else:
  DEVICE="cpu"

model = deeplabv3p.DeepLabV3_Plus(num_classes=cfg.n_classes)

if len(DEVICE)>1:
  model = nn.DataParallel(model)
else:
  model = model.cuda()

# train dataset
train_dataset = dataloader.SegmentationDataset(
        img_dir = '',
        mask_dir = '',
        txt = '',
        shape=(cfg.HEIGHT, cfg.WIDTH),
        classes = cfg.n_classes
    )

# valid dataset
valid_dataset = dataloader.SegmentationDataset(
        img_dir = '',
        mask_dir = '',
        txt = '',
        shape=(cfg.HEIGHT, cfg.WIDTH),
        classes = cfg.n_classes
    )

# dataloader
train_loader = DataLoader(train_dataset, batch_size=cfg.batch, shuffle=True, num_workers=16, drop_last=True, collate_fn=dataloader.collate_fn)
val_loader = DataLoader(valid_dataset, batch_size=cfg.batch, shuffle=False, num_workers=16, collate_fn=dataloader.collate_fn)

#define loss
criterion = loss.DiceFocalLoss()

#define optimizer
optimizer = torch.optim.AdamW(params=model.parameters(), lr=cfg.lr, weight_decay=0.009)

#define scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.09, patience=4*cfg.val_term)

def train(model, train_loader, valid_loader, epoch, val_term):
    for i in range(epoch):

      model.train()

      ious = 0
      epoch_loss = 0

      for batch_idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        #len(train_loader) = iterations per epoch
        data, target = data.cuda(), target.cuda()
        output = model(data)
        target = target.permute(0,3,1,2)

        loss_output = criterion(output.float(), target.float())
        loss_output.backward()

        #calculate metrics
        iou_score = iou(output.int(), target.int(), cfg.n_classes) #iteration 한번에 대한 class iou 값
        ious += iou_score #ious : iteration 별 class iou의 합 = epoch 1회의 class iou(평균값 아님)
        iter_loss = loss_output.item()
        epoch_loss += iter_loss
        if batch_idx%(len(train_loader)//4) == 0:
          wandb.log({"loss": iter_loss})

          for j in range(cfg.n_classes):
            wandb.log({f"iou_class{j}": iou_score[j]})
          wandb.log({"miou": torch.nanmean(iou_score)})
      epoch_ious = ious/len(train_loader) #epoch 1회의 class iou
      epoch_miou = torch.nanmean(epoch_ious) #epoch 1회의 miou
      epoch_loss = epoch_loss/len(train_loader) #=loss_output.mean().item()/cfg.batch, epoch 당 이미지 1개에 대한 loss 값

      print(f"Epoch {i}/{epoch}\nLoss: {epoch_loss:.6f}\nclass IoU: {epoch_ious.numpy()}\nmIoU: {epoch_miou.numpy():.6f}")

      if i % val_term == val_term-1 or i==0:
        val_loss = validation(model, valid_loader)
        torch.save(model.state_dict(), 'weights/epoch_{}.pth'.format(i))
      optimizer.step()
      scheduler.step(val_loss)


def validation(model, valid_loader):
    model.eval()
    valid_loss = 0
    ious = 0

    sample = None #for visualization

    with torch.no_grad():
        for _, (data, target) in tqdm(enumerate(valid_loader), total=len(valid_loader)): #per batch
            data, target = data.cuda(), target.cuda()
            #data : N, C, H, W
            output = model(data)
            target = target.permute(0,3,1,2) #N, H, W, C -> N, C, H, W
            sample = output
            valid_loss += criterion(output.float(), target.float()) #iteration loss, valid_loss: epoch loss
            batch_ious = iou(output.int(), target.int(), cfg.n_classes) #iteration class iou
            ious += batch_ious
    ious = ious/len(valid_loader) #전체 class별 iou
    sample = data[0].permute(1,2,0).cpu().numpy()
    predict_sample = torch.argmax(output[0], dim=0).cpu().numpy()

    valid_loss /= len(valid_loader)
    miou = torch.nanmean(ious)
    print(f"Validation\nLoss: {valid_loss:.6f}\nclass IoU: {ious.numpy()}\nmIoU: {miou.numpy():.6f}")

    wandb.log({"val_loss": valid_loss, "val_miou": miou})

    for j in range(cfg.n_classes):
        wandb.log({f"val_iou_class{j}": ious[j]})
    wandb.log({"val_sample":wandb.Image(sample, masks={"predictions": {"mask_data": predict_sample, "class_labels": cfg.CLASSES}})})
    return valid_loss

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.6f}".format(x)})
    wandb.init(project='dlv3p_custom')
    wandb.config.update({"batch_size": cfg.batch, "learning_rate":cfg.lr, "epochs": cfg.EPOCHS})

    wandb.watch(model)
    train(model=model, train_loader=train_loader, valid_loader=val_loader, epoch = cfg.EPOCHS, val_term=cfg.val_term)