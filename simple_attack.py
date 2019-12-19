# Helper function for extracting features from pre-trained models
import sys, os
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np
import glob

from attacker import Attacker
from loader import ImageNet_A
from utils.Resnet import resnet152_denoise, resnet101_denoise, resnet152
from utils.Normalize import Normalize, Permute


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='./results/', type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--steps', default=60, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--init_norm', default=1, type=float)
    parser.add_argument('--max_norm', default=32, type=float)
    parser.add_argument('--min_loss', default=0, type=float)
    parser.add_argument('--targeted', action='store_true')
    args = parser.parse_args()

    pretrained_model1 = resnet101_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Denoise_Resnext101.pytorch'))
    pretrained_model1.load_state_dict(loaded_state_dict, strict=True)

    pretrained_model2 = resnet152_denoise()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Denoise_Resnet152.pytorch'))
    pretrained_model2.load_state_dict(loaded_state_dict)

    pretrained_model3 = resnet152()
    loaded_state_dict = torch.load(os.path.join('weight', 'Adv_Resnet152.pytorch'))
    pretrained_model3.load_state_dict(loaded_state_dict)

    model1 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model1
        )
    
    model2 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model2
        )

    model3 = nn.Sequential(
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            Permute([2, 1, 0]),
            pretrained_model3
        )

    model1.cuda()
    model1.eval()

    model2.cuda()
    model2.eval()

    model3.cuda()
    model3.eval()

    output_dir = os.path.join(args.output_dir, 'images')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    dataset = ImageNet_A(args.input_dir, targeted=args.targeted)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=(1 if (args.min_loss and (not args.targeted)) else args.batch_size), 
                                         shuffle=False)

    attacker = Attacker(steps=args.steps, 
                        gamma=args.gamma, 
                        init_norm=args.init_norm, 
                        max_norm=args.max_norm/255.0, 
                        min_loss=(args.min_loss if (args.min_loss and (not args.targeted)) else None), 
                        device=torch.device('cuda'))

    for ind, (img, label, filenames) in enumerate(loader):
        img_g, label_g = img.cuda(), label.cuda()

        adv = attacker.attack(model1, model2, model3, img_g, labels=label_g, targeted=args.targeted)

        for bind, filename in enumerate(filenames):
            out_img = adv[bind].detach().cpu().numpy()
            delta_img = np.abs(out_img - img[bind].numpy()) * 255.0

            print('Attack on {}:'.format(os.path.split(filename)[-1]))
            print('Max: {0:.0f}, Mean: {1:.2f}'.format(
                (np.max(delta_img)),
                (np.mean(delta_img)) 
            ))

            out_img = np.transpose(out_img, axes=[1, 2, 0]) * 255.0
            out_img = out_img[:, :, ::-1]

            out_filename = os.path.join(output_dir, os.path.split(filename)[-1])
            cv2.imwrite(out_filename, out_img)