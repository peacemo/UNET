'''
 # @ Author: Chen Xueqiang
 # @ Create Time: 2023-10-08 10:51:48
 # @ Modified by: Chen Xueqiang
 # @ Modified time: 2023-10-08 10:51:54
 # @ Description: training file for 'Carvana' dataset. 
 '''


import torch 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import argparse
from utils import get_loaders, save_checkpoint, load_checkpoint, validation

# Hyperparameters
parser = argparse.ArgumentParser(description='Train')

parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs')

parser.add_argument('--device', type=str, default='cpu', help='train device, cuda or cpu')
parser.add_argument('--num_workers', type=int, default=2, help='number of workers')
parser.add_argument('--pin_memory', type=bool, default=True, help='pin memory')
parser.add_argument('--load_model', type=bool, default=False, help='load model')

parser.add_argument('--image_dir', type=str, default='/home/shikamaru/datasets/carvana-image-masking-challenge/train', help='image directory')
parser.add_argument('--mask_dir', type=str, default='/home/shikamaru/datasets/carvana-image-masking-challenge/train_masks', help='mask directory')
parser.add_argument('--image_height', type=int, default=160, help='image height')
parser.add_argument('--image_width', type=int, default=240, help='image width')

args = parser.parse_args()

if torch.cuda.is_available():
    args.device = 'cuda'

def train(dataloader, model, optimizer, loss_fn, scaler):
    loop = tqdm(dataloader)

    for batch_idx, (data, masks) in enumerate(loop):
        data = data.to(device=args.device)
        masks = masks.float().unsqueeze(1).to(device=args.device)  # float() for BCE loss, unsqueeze() adds the channel dimension. 

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, masks)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm
        loop.set_postfix(loss=loss.item())
    pass

def main():
    transform = A.Compose(
        [
            A.Resize(height=args.image_height, width=args.image_width),
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.3), 
            # A.VerticalFlip(p=0.1), 
            A.Normalize(
                mean=[0.0, 0.0, 0.0], 
                std=[1.0, 1.0, 1.0], 
                max_pixel_value=255.0
            ), 
            ToTensorV2()
        ]
    )

    model = UNET(in_channels=3, out_channels=1, features=[4,8,16,32]).to(device=args.device)
    loss = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = get_loaders(
        train_dir=args.image_dir, 
        mask_dir=args.mask_dir, 
        batch_size=args.bs, 
        transform=transform, 
        num_workers=args.num_workers, 
        pin_memory=args.pin_memory, 
        val_size=0.2
        )
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.epochs):
        train(train_loader, model, optimizer, loss, scaler)

        # save checkpoint
        checkpoint = {
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # validation
        validation(model, val_loader)


    pass

main()
