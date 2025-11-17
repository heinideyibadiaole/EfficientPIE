"""

@ Description:
@ Project:APCIL
@ Author:qufang
@ Create:2024/6/20 20:57

"""
import argparse
import os
import numpy as np
from thop import profile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.pie_data import PIE
from utils.my_dataset import MyDataSet
from models.EfficientPIE_backup import EfficientPIE
from utils.train_val import evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)

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
                 'output_type': ['intention_binary']
                 }

    data_type = {'encoder_input_type': data_opts['encoder_input_type'],
                 'decoder_input_type': data_opts['decoder_input_type'],
                 'output_type': data_opts['output_type']}

    # get the dataset api
    PIE_dataset = PIE(data_path=args.data_path)
    # generate_data_trajectory_sequence() will call _get_intention() to generate intention data for train set
    test_seq = PIE_dataset.generate_data_trajectory_sequence('test', **data_opts)
    seq_length = data_opts['max_size_observe']
    test_seq_for_dataset = PIE_dataset.get_train_val_data(test_seq, data_type, seq_length, data_opts['seq_overlap_rate'])

    # define the transform
    data_transform = {
        "test": transforms.Compose([transforms.Resize([300, 300]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                    ])
    }
    # define dataset, for dataloader
    test_dataset = MyDataSet(images_seq=test_seq_for_dataset, data_opts=data_opts, transform=data_transform['test'])
    # set the training parameters
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # dataloader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, pin_memory=True,
                             num_workers=nw, collate_fn=test_dataset.collate_fn)

    model = EfficientPIE(num_classes=2).to(device)
    print(model)
    # load model weights
    version = args.version
    model_weight_path = f"./weights_v{version}/model_8_PIE_IL_step14_new.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # test
    print("using the weight:{}".format(model_weight_path))
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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--version', type=int, default=8)
    # PIE dataset path
    parser.add_argument('--data-path', type=str,
                        default="/home/yphe/FangQu_temporary/PIEDataset")  # absolute path
    parser.add_argument('--device', default='cuda:6', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)
