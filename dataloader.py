import torch
from PIL import Image
import numpy as np
from torch.utils import data
from PIL import Image
import os
from torchvision import transforms

tfs = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
  ])

def collate_fn(batch):
    images, labels = zip(*batch)
    images = [torch.from_numpy(np.array(image)) for image in images]
    images = torch.stack(images)
    labels = [torch.from_numpy(np.array(label)) for label in labels]
    labels = torch.stack(labels)
    return images, labels

#VOC file list loader
def filelists(txtpath):
    filelist = []
    with open(os.path.join(txtpath), 'r') as f:
        for line in f:
            filelist.append(line.strip())
    return filelist

#segmentation dataloader
class SegmentationDataset(data.Dataset):
    def __init__(self, img_dir, mask_dir, txt, transform=tfs, classes=None, shape = (512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.classes = classes
        self.shape = shape
        filelist = []
        with open(os.path.join(txt), 'r') as f:
            for line in f:
                filelist.append(line.strip())
        self.ids = filelist
        self.ids.sort()

    def __getitem__(self, index):
        #load image and mask
        img_path = os.path.join(self.img_dir, self.ids[index]+'.jpg')
        mask_path = os.path.join(self.mask_dir, self.ids[index]+'.png')
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        img = img.resize(self.shape, Image.BILINEAR)
        mask = mask.resize(self.shape, Image.BILINEAR)

        #transform
        if self.transform:
          img = self.transform(img)

        mask = np.array(mask)
        mask = np.where(mask>self.classes-1, 0, mask)
        mask = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), num_classes=self.classes).numpy()

        return img, mask

    def __len__(self):
        return len(self.ids)