"""

@ Description:
@ Project:APCIL
@ Author:qufang
@ Create:2024/7/22 12:58

"""
import argparse
import os

import numpy as np
import torch
import torchvision.models
from thop import profile
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet
from models.EfficientPIE import EfficientPIE
from utils.train_val import evaluate


def main(args):

    data_opts = {'fstride': 1,  # The stride specifies the sampling resolution, i.e. every nth frame is used for processing.
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')], # use pedestrains within 0px - infinite px
                 # This parameter can be used to fix the aspect ratio (width/height) of bounding boxes.
                 'squarify_ratio': 0,  # 0 means the original bounding boxes are returned.
                 'data_split_type': 'random',  # kfold, random, default
                 'seq_type': 'intention',  # crossing , intention
                 'min_track_size': 0,  # discard tracks that are shorter
                 'max_size_observe': 15,  # number of observation frames
                 # 'max_size_predict': 5,  # number of prediction frames, no use
                 'seq_overlap_rate': 0.5,  # how much consecutive sequences overlap
                 'balance': True,  # balance the training and testing samples
                 'crop_type': 'context',  # crop 2x size of bbox around the pedestrian
                 'crop_mode': 'pad_resize',  # pad with 0s and resize to VGG input
                 'encoder_input_type': [],
                 'decoder_input_type': ['bbox'],
                 'output_type': ['intent']
                 }

    data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                 'decoder_input_type': data_opts['decoder_input_type'],
                 'output_type': data_opts['output_type']}

    # get the dataset api
    JAAD_dataset = JAAD(data_path=args.data_path)
    # generate_data_trajectory_sequence() will call _get_intention() to generate intention data for train set
    test_seq = JAAD_dataset.generate_data_trajectory_sequence('test', **data_opts)
    seq_length = data_opts['max_size_observe']
    test_seq_for_dataset = JAAD_dataset.get_train_val_data(test_seq, data_type, seq_length, data_opts['seq_overlap_rate'])

    # define the transform
    data_transform = {
        "test": transforms.Compose([transforms.Resize([300, 300]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    }

    # define dataset, for dataloader
    test_dataset = MyDataSet(images_seq=test_seq_for_dataset, data_opts=data_opts, transform=data_transform['test'])

    # set parameters
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=nw, collate_fn=test_dataset.collate_fn)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = EfficientPIE(num_classes=2).to(device)
    print(model)
    # load model weights
    model_weight_path = "./weights/transfer_noisy_model_JAAD.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # test accuracy
    print("using the weight:{}".format(model_weight_path))
    print("test set length:{}".format(test_dataset.__len__()))
    print(args)
    print("Start Testing!")
    evaluate(model=model, dataloader=test_loader, device=device, epoch=0)

    # test inference speed,Flop and Params
    dummy_imgs = []
    for i in range(args.batch_size):
        img, label = test_dataset.__getitem__(i)
        dummy_imgs.append(torch.as_tensor(img))
    dummy_imgs = torch.stack(dummy_imgs).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros((repetitions, 1))

    # gpu warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_imgs)

    # measure
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_imgs)
            ender.record()
            # synchronize
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
        # compute the inference time
        timings /= args.batch_size
        mean_time = np.mean(timings)
        std_time = np.std(timings)
        print("Mean Inference Time: {:.4f} ms".format(mean_time))
        print("Standard Deviation: {:.4f} ms".format(std_time))
        flops, params = profile(model, (dummy_imgs,))
        print('flops: %.2f M, params: %.2f M' % (flops / 1e6 / args.batch_size, params / 1e6))

    print("Finished!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data-path', type=str,
                        default="/home/yphe/FangQu_temporary/JAAD")  # absolute path
    parser.add_argument('--device', default='cuda:2', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)