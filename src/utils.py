import torch 
import torchvision
from dataset import CarvanaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print('=> Saving checkpoint...')
    torch.save(state, filename)
    print('=> Checkpoint saved. ')

def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoint...')
    model.load_state_dict(checkpoint['state_dict'])
    print('=> Checkpoint loaded. ')

def get_loaders(
        train_dir, 
        mask_dir, 
        batch_size, 
        transform, 
        num_workers=4, 
        pin_memory=True, 
        val_size=0.2
):
    dataset = CarvanaDataset(image_dir=train_dir, mask_dir=mask_dir, transform=transform)

    train_size = int((1-val_size) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split (dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def dice_coef(y_true, y_pred): 
    y_pred = (y_pred > 0.5)
    y_true = y_true == 1.0

    inter = (y_true & y_pred).sum()
    den = y_true.sum() + y_pred.sum()
    dice = 2 * inter / (den + 1e-8)
    return dice.item()

def iou_coef(y_true, y_pred):
    y_pred = (y_pred > 0.5)
    y_true = y_true == 1.0

    inter = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    iou = inter / (union + 1e-8)
    return iou.item()

def validation(model, val_loader, device='cuda'):
    loop = tqdm(val_loader)

    model.eval()
    dice = []
    iou = []

    with torch.no_grad():
        for data, masks in loop: 
            data = data.to(device)
            masks = masks.to(device)
            preds = model(data)
            preds = (preds > 0.5)
            masks = (masks == 1.0)
            dice.append(dice_coef(masks, preds))
            iou.append(iou_coef(masks, preds))

    print(f'Average DICE: {np.mean(dice)}')
    print(f'Average IoU: {np.mean(iou)}')

    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
    