"""

@ Description:
@ Project:APCIL
@ Author:qufang
@ Create:2024/6/20 23:11

"""
import argparse
import os

import torch
import torchvision.datasets
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from models.EfficientPIE_backup import EfficientPIE
from utils.train_val import pre_train_one_epoch, pre_evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./pre_train_weights_efficientpie") is False:
        os.makedirs("./pre_train_weights_efficientpie")

    # define the transform
    data_transform = {
        "train": transforms.Compose([transforms.Resize([300, 300]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ]),
        "val": transforms.Compose([transforms.Resize([300, 300]),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                   ])
    }
    # define dataset, for dataloader
    train_dataset = torchvision.datasets.ImageFolder(root=args.train_path, transform=data_transform['train'])
    val_dataset = torchvision.datasets.ImageFolder(root=args.val_path, transform=data_transform['val'])

    # set the training parameters
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw)

    model = EfficientPIE(num_classes=1000).to(device)
    model_weight_path = "./pre_train_weights_efficientpie/min_loss_pretrained_model_imagenet_backup.pth"
    weights_dict = torch.load(model_weight_path, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
    print(model.load_state_dict(load_weights_dict, strict=False))

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)

    # train and validate
    best_val_acc = 0.0
    min_loss = 100.0
    save_path = "./pre_train_weights_efficientpie/best_pretrained_model_imagenet.pth"
    min_loss_path = "./pre_train_weights_efficientpie/min_loss_pretrained_model_imagenet.pth"
    print(model)
    print("Start Training now!")
    for epoch in range(args.epochs):
        train_loss, train_acc = pre_train_one_epoch(model=model, optimizer=optimizer, dataloader=train_loader,
                                                    device=device, epoch=epoch)
        scheduler.step()  # learning rate will be changed
        val_loss, val_acc = pre_evaluate(model=model, dataloader=val_loader, device=device, epoch=epoch)
        tags = ["train_loss", "train_acc",
                "val_loss", "val_acc",
                "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model at epoch {epoch} with validation accuracy: {val_acc:.4f}')
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), min_loss_path)
            print(f'Saved min loss model at epoch {epoch} with validation accuracy: {val_acc:.4f}, loss: {val_loss:.4f}')

    print("Finished Training!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=500)
    # PIE dataset path
    parser.add_argument('--train_path', type=str, default="/home/fqu/FangQu_temporary/imagenet/train")
    parser.add_argument('--val_path', type=str, default="/home/fqu/FangQu_temporary/imagenet/val")
    parser.add_argument('--device', default='cuda:6', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)