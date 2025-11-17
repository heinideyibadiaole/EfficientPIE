"""

@ Description:
@ Project:APCIL
@ Author:qufang
@ Create:2024/6/20 20:57

"""
import argparse
import os
import re

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.pie_data import PIE
from utils.my_dataset import MyDataSet
from models.EfficientPIE_backup import EfficientPIE
from utils.train_val import train_one_epoch, evaluate, incremental_learning_train


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    version = args.version
    if os.path.exists(f"./weights_v{version}") is False:
        os.makedirs(f"./weights_v{version}")

    data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'random',
                 'seq_type': 'intention',
                 'min_track_size': 0,
                 'max_size_observe': 15,
                 'seq_overlap_rate': 0.5,
                 'balance': True,
                 'crop_type': 'context',
                 'crop_mode': 'pad_resize',
                 'encoder_input_type': [],
                 'decoder_input_type': ['bbox'],
                 'output_type': ['intention_binary']
                 }

    data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                 'decoder_input_type': data_opts['decoder_input_type'],
                 'output_type': data_opts['output_type']}
    PIE_dataset = PIE(data_path=args.data_path)
    train_seq_unbalanced = PIE_dataset.generate_data_trajectory_sequence('train', **data_opts)
    train_seq = train_seq_unbalanced
    val_seq_unbalanced = PIE_dataset.generate_data_trajectory_sequence('val', **data_opts)
    val_seq = val_seq_unbalanced
    seq_length = data_opts['max_size_observe']

    train_seq_for_dataset = PIE_dataset.get_train_val_data(train_seq, data_type, seq_length,
                                                           data_opts['seq_overlap_rate'])
    val_seq_for_dataset = PIE_dataset.get_train_val_data(val_seq, data_type, seq_length, data_opts['seq_overlap_rate'])

    data_transform = {
        "train": transforms.Compose([transforms.Resize([300, 300]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                     ]),
        "val": transforms.Compose([transforms.Resize([300, 300]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                   ])
    }
    # define dataset, for dataloader
    train_dataset = MyDataSet(images_seq=train_seq_for_dataset, data_opts=data_opts, transform=data_transform['train'], step=args.step)
    val_dataset = MyDataSet(images_seq=val_seq_for_dataset, data_opts=data_opts, transform=data_transform['val'], step=args.step)
    # set the training parameters
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, pin_memory=True,
                              num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, pin_memory=True,
                            num_workers=nw, collate_fn=val_dataset.collate_fn)

    model = EfficientPIE(num_classes=2).to(device)
    # only load the feature extraction layers of prev model
    weights_dict = torch.load(args.prev_weights, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items() if k in model.state_dict()
                         and model.state_dict()[k].numel() == v.numel()
                         and not k.startswith('classifier')
                         }
    print(model.load_state_dict(load_weights_dict, strict=False))

    # load the model of prev step
    prev_model = EfficientPIE(num_classes=2).to(device)
    weights_dict = torch.load(args.prev_weights, map_location=device)
    load_weights_dict = {k: v for k, v in weights_dict.items()
                         if prev_model.state_dict()[k].numel() == v.numel()}
    print(prev_model.load_state_dict(load_weights_dict, strict=False))

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.RMSprop(pg, lr=args.lr, weight_decay=0.0001)
    # optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-7)

    # train and validate
    best_val_acc = 0.0
    min_loss = 100.0
    save_path = f"./weights_v{version}/best_model_PIE_IL_step{args.step}_new.pth"
    min_loss_path = f"./weights_v{version}/min_loss_PIE_IL_step{args.step}_new.pth"
    print(model)
    print("Start Training now!")
    for epoch in range(args.epochs):
        train_loss, train_acc, train_precision, \
        train_recall, train_f1 = incremental_learning_train(model=model,
                                                            optimizer=optimizer,
                                                            dataloader=train_loader,
                                                            device=device,
                                                            epoch=epoch,
                                                            prev_model=prev_model)
        scheduler.step()  # learning rate will be changed
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(model=model,
                                                                        dataloader=val_loader,
                                                                        device=device,
                                                                        epoch=epoch)
        tags = ["train_loss", "train_acc",
                "val_loss", "val_acc",
                "learning_rate",
                "train_precision", "train_recall", "train_f1",
                "val_precision", "val_recall", "val_f1"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar(tags[5], train_precision, epoch)
        tb_writer.add_scalar(tags[6], train_recall, epoch)
        tb_writer.add_scalar(tags[7], train_f1, epoch)
        tb_writer.add_scalar(tags[8], val_precision, epoch)
        tb_writer.add_scalar(tags[9], val_recall, epoch)
        tb_writer.add_scalar(tags[10], val_f1, epoch)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f'Saved best model at epoch {epoch} with validation accuracy: {val_acc:.4f}')
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save(model.state_dict(), min_loss_path)
            print(
                f'Saved min loss model at epoch {epoch} with validation accuracy: {val_acc:.4f}, loss: {val_loss:.4f}')
        if epoch >= 5:
            torch.save(model.state_dict(), f"./weights_v{version}/model_{epoch}_PIE_IL_step{args.step}_new.pth")
            print(f'Saved model at epoch {epoch} with validation accuracy: {val_acc:.4f}, loss: {val_loss:.4f}')

    print("Finished Training!")

# step   0        2        4       6       8       10      12      14
# ori   90.6     89.4     90.9    90.3    90.1    90.1    90.1    91.3
# epoch  25       9        10    11 13     14      6       6       15
# after 90.6     91.2     91.0    91.3    91.2    91.7    91.7    91.7
# every training, change the step, prev_weights and reverse_step in "my_dataset.py"
# make sure reverse_step + step = 15
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--version', type=int, default=8)
    parser.add_argument('--step', type=int, default=14)
    parser.add_argument('--prev_weights', type=str, default="weights_v8/model_6_PIE_IL_step12.pth")
    # PIE dataset path
    parser.add_argument('--data-path', type=str,
                        default="/home/yphe/FangQu_temporary/PIEDataset")  # absolute path
    parser.add_argument('--weights', type=str,
                        default="pre_train_weights_efficientpie/min_loss_pretrained_model_imagenet_backup.pth",
                        help='initial weights path')
    # parser.add_argument('--weights', type=str, default="",
    #                     help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
